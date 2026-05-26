# -*- coding: utf-8 -*-
"""Workchain to run a Quantum ESPRESSO pw.x calculation with automated error handling and restarts."""

from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import BaseRestartWorkChain, calcfunction, while_
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_quantumespresso.common.types import ElectronicType, SpinType
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin
from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import (
    create_kpoints_from_distance,
)
from aiida_mechanical.tools.structures import (
    get_standardized_structure_pymatgen,
    convert_standardized_structure_pymatgen_to_qe,
)
from aiida_mechanical.calculations.thermo_pw import Thermo_pwCalculation


class Thermo_pwBaseWorkChain(ProtocolMixin, BaseRestartWorkChain):
    """Workchain to run a thermo_pw.x calculation with automated error handling and restarts."""

    # pylint: disable=too-many-public-methods, too-many-statements

    _process_class = Thermo_pwCalculation

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        # yapf: disable
        super().define(spec)
        spec.expose_inputs(Thermo_pwCalculation, namespace='thermo_pw', exclude=('kpoints',))
        spec.input('kpoints', valid_type=orm.KpointsData, required=False,
            help='An explicit k-points list or mesh. Either this or `kpoints_distance` has to be provided.')
        spec.input('kpoints_distance', valid_type=orm.Float, required=False,
            help='The minimum desired distance in 1/Å between k-points in reciprocal space. The explicit k-points will '
                 'be generated automatically by a calculation function based on the input structure.')
        spec.input('kpoints_force_parity', valid_type=orm.Bool, required=False,
            help='Optional input when constructing the k-points based on a desired `kpoints_distance`. Setting this to '
                 '`True` will force the k-point mesh to have an even number of points along each lattice vector except '
                 'for any non-periodic directions.')

        spec.outline(
            cls.setup,
            cls.validate_kpoints,
            while_(cls.should_run_process)(
                cls.prepare_process,
                cls.run_process,
                cls.inspect_process,
            ),
            cls.results,
        )

        spec.expose_outputs(Thermo_pwCalculation)

        spec.exit_code(201, 'ERROR_INVALID_INPUT_PSEUDO_POTENTIALS',
            message='The explicit `pseudos` or `pseudo_family` could not be used to get the necessary pseudos.')
        spec.exit_code(202, 'ERROR_INVALID_INPUT_KPOINTS',
            message='Neither the `kpoints` nor the `kpoints_distance` input was specified.')
        spec.exit_code(203, 'ERROR_INVALID_INPUT_RESOURCES',
            message='Neither the `options` nor `automatic_parallelization` input was specified. '
                    'This exit status has been deprecated as the check it corresponded to was incorrect.')
        spec.exit_code(204, 'ERROR_INVALID_INPUT_RESOURCES_UNDERSPECIFIED',
            message='The `metadata.options` did not specify both `resources.num_machines` and `max_wallclock_seconds`. '
                    'This exit status has been deprecated as the check it corresponded to was incorrect.')
        spec.exit_code(210, 'ERROR_INVALID_INPUT_AUTOMATIC_PARALLELIZATION_MISSING_KEY',
            message='Required key for `automatic_parallelization` was not specified.'
                    'This exit status has been deprecated as the automatic parallellization feature was removed.')
        spec.exit_code(211, 'ERROR_INVALID_INPUT_AUTOMATIC_PARALLELIZATION_UNRECOGNIZED_KEY',
            message='Unrecognized keys were specified for `automatic_parallelization`.'
                    'This exit status has been deprecated as the automatic parallellization feature was removed.')
        spec.exit_code(300, 'ERROR_UNRECOVERABLE_FAILURE',
            message='The calculation failed with an unidentified unrecoverable error.')
        spec.exit_code(310, 'ERROR_KNOWN_UNRECOVERABLE_FAILURE',
            message='The calculation failed with a known unrecoverable error.')
        # yapf: enable

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files

        import aiida_mechanical.workflows.protocols as thermo_pw_protocols

        return files(thermo_pw_protocols) / "base.yaml"

    @classmethod
    def get_builder_from_protocol(
        cls,
        code,
        structure,
        protocol=None,
        overrides=None,
        electronic_type=ElectronicType.METAL,
        spin_type=SpinType.NONE,
        initial_magnetic_moments=None,
        options=None,
        **_,
    ):
        inputs = cls.get_protocol_inputs(protocol, overrides)

        from copy import deepcopy

        pw_overrides = deepcopy(inputs)
        pw_overrides["pw"] = pw_overrides.pop("thermo_pw")
        pw_overrides["pw"].pop("thermo_control")
        pw_builder = PwBaseWorkChain.get_builder_from_protocol(
            code=code,
            structure=structure,
            protocol=protocol,
            overrides=pw_overrides,
            electronic_type=electronic_type,
            spin_type=spin_type,
            initial_magnetic_moments=initial_magnetic_moments,
            options=options,
        )

        builder = cls.get_builder()

        builder.thermo_pw._data = pw_builder.pw._data
        builder.thermo_pw["thermo_control"] = inputs["thermo_pw"]["thermo_control"]

        builder.clean_workdir = orm.Bool(inputs["clean_workdir"])
        if "kpoints" in inputs:
            builder.kpoints = inputs["kpoints"]
        else:
            builder.kpoints_distance = orm.Float(inputs["kpoints_distance"])
        builder.kpoints_force_parity = orm.Bool(inputs["kpoints_force_parity"])
        builder.max_iterations = orm.Int(inputs["max_iterations"])

        return builder

    @staticmethod
    @calcfunction
    def format_structure(structure: orm.StructureData):
        """Format the structure to the QE convention."""
        extras = structure.base.extras.all
        standardized_pym_structure = get_standardized_structure_pymatgen(structure)

        qe_pym_structure, structure_parameters = (
            convert_standardized_structure_pymatgen_to_qe(standardized_pym_structure)
        )
        formated_structure = orm.StructureData(pymatgen_structure=qe_pym_structure)

        formated_structure.base.extras.set_many(extras)
        return {
            "formated_structure": formated_structure,
            "structure_parameters": orm.Dict(dict=structure_parameters),
        }

    def setup(self):
        """Call the ``setup`` of the ``BaseRestartWorkChain`` and create the inputs dictionary in ``self.ctx.inputs``.

        This ``self.ctx.inputs`` dictionary will be used by the ``BaseRestartWorkChain`` to submit the calculations
        in the internal loop.

        The ``parameters`` and ``settings`` input ``Dict`` nodes are converted into a regular dictionary and the
        default namelists for the ``parameters`` are set to empty dictionaries if not specified.
        """
        super().setup()
        self.ctx.inputs = AttributeDict(
            self.exposed_inputs(Thermo_pwCalculation, "thermo_pw")
        )

        results = self.format_structure(self.ctx.inputs.structure)
        self.ctx.inputs.structure = results["formated_structure"]

        parameters = self.ctx.inputs.parameters.get_dict()
        parameters["SYSTEM"]["ibrav"] = results["structure_parameters"]["ibrav"]
        self.ctx.inputs.parameters = orm.Dict(dict=parameters)

        self.ctx.inputs.settings = (
            self.ctx.inputs.settings.get_dict() if "settings" in self.ctx.inputs else {}
        )

    def validate_kpoints(self):
        """Validate the inputs related to k-points.

        Either an explicit `KpointsData` with given mesh/path, or a desired k-points distance should be specified. In
        the case of the latter, the `KpointsData` will be constructed for the input `StructureData` using the
        `create_kpoints_from_distance` calculation function.
        """
        if all(key not in self.inputs for key in ["kpoints", "kpoints_distance"]):
            return self.exit_codes.ERROR_INVALID_INPUT_KPOINTS

        try:
            kpoints = self.inputs.kpoints
        except AttributeError:
            inputs = {
                "structure": self.inputs.thermo_pw.structure,
                "distance": self.inputs.kpoints_distance,
                "force_parity": self.inputs.get(
                    "kpoints_force_parity", orm.Bool(False)
                ),
                "metadata": {"call_link_label": "create_kpoints_from_distance"},
            }
            kpoints = create_kpoints_from_distance(**inputs)  # pylint: disable=unexpected-keyword-arg

        self.ctx.inputs.kpoints = kpoints

    def prepare_process(self):
        pass

    @staticmethod
    def _clean_workdir(node):
        """Clean the working directories of all child calculations if `clean_workdir=True` in the inputs."""

        cleaned_calcs = []

        for called_descendant in node.called_descendants:
            if isinstance(called_descendant, orm.CalcJobNode):
                try:
                    called_descendant.outputs.remote_folder._clean()  # pylint: disable=protected-access
                    cleaned_calcs.append(called_descendant.pk)
                except (IOError, OSError, KeyError):
                    pass

        return cleaned_calcs

    def on_terminated(self):
        """Clean the working directories of all child calculations if `clean_workdir=True` in the inputs."""
        super().on_terminated()

        if self.inputs.clean_workdir.value is False:
            self.report("remote folders will not be cleaned")
            return

        if self.node.is_finished_ok:
            cleaned_calcs = self._clean_workdir(self.node)

            if cleaned_calcs:
                self.report(
                    f"cleaned remote folders of calculations: {' '.join(map(str, cleaned_calcs))}"
                )
