# -*- coding: utf-8 -*-
"""`CalcJob` implementation for the pw.x code of Quantum ESPRESSO."""

from aiida import orm
from aiida.common import exceptions

from aiida_quantumespresso.calculations.pw import PwCalculation
from aiida_quantumespresso.calculations import _uppercase_dict

from aiida_quantumespresso.utils.convert import convert_input_to_namelist_entry


class Thermo_pwCalculation(PwCalculation):
    """
    Base class for Thermo_pw calculations.
    Thermo_pw share the same input file as PwCalculation.
    We directly reuse the definition of PwCalculation with an extra thermo_control input.
    """

    _DEFAULT_THERMO_CONTROL = "thermo_control"
    _OUTPUT_ELASTIC_CONSTANTS_SUBFOLDER = "./elastic_constants/"
    _OUTPUT_GNUPLOT_FILES_SUBFOLDER = "./gnuplot_files/"
    _OUTPUT_THERM_FILES_SUBFOLDER = "./therm_files/"
    _OUTPUT_ELASTIC_CONSTANTS_FILE = "output_el_cons.dat.g1"
    _OUTPUT_THERM_DEBYE_FILE = "output_therm.dat_debye.g1"

    _COMPULSORY_NAMELISTS = [
        "INPUT_THERMO",
    ]

    _ENABLED_KEYWORDS = [
        ("INPUT_THERMO", "what"),
        ("INPUT_THERMO", "find_ibrav"),
        ("INPUT_THERMO", "frozen_ions"),
    ]

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input(
            "code",
            valid_type=orm.Code,
            help="The thermo_pw.x code to run the calculation.",
        )

        spec.input(
            "thermo_control",
            valid_type=orm.Dict,
            help="The parameters for thermo_control file.",
        )

        spec.inputs["metadata"]["options"]["parser_name"].default = "thermo_pw"
        spec.output(
            "elastic_constants",
            valid_type=orm.ArrayData,
            required=False,
            help="The elastic constants.",
        )

        spec.output(
            "therm_dat_debye",
            valid_type=orm.XyData,
            required=False,
            help="The thermal properties.",
        )

        # exit codes reported by thermo_pw.x starts from 8
        spec.exit_code(
            801, "ERROR_LATGEN", message=("This error usually happens when ibrav = 0")
        )

    def prepare_for_submission(self, folder):
        # Reuse the prepare_for_submission method of PwCalculation

        calcinfo = super().prepare_for_submission(folder)

        if "settings" in self.inputs:
            settings = _uppercase_dict(
                self.inputs.settings.get_dict(), dict_name="settings"
            )
        else:
            settings = {}

        thermo_control = self.inputs.thermo_control.get_dict()

        for flag in thermo_control.keys():
            if ("INPUT_THERMO", flag) not in self._ENABLED_KEYWORDS:
                raise exceptions.InputValidationError(
                    f"'{flag}' flag is not enabled for now."
                )

        with folder.open(self._DEFAULT_THERMO_CONTROL, "w") as handle:
            handle.write("&INPUT_THERMO\n")
            for key, value in thermo_control.items():
                handle.write(convert_input_to_namelist_entry(key, value))
            handle.write("/\n")

        cmdline_params = self._add_parallelization_flags_to_cmdline_params(
            cmdline_params=settings.pop("CMDLINE", [])
        )

        calcinfo.retrieve_list.append(self._DEFAULT_THERMO_CONTROL)
        calcinfo.retrieve_list.append(
            self._OUTPUT_ELASTIC_CONSTANTS_SUBFOLDER
            + self._OUTPUT_ELASTIC_CONSTANTS_FILE
        )
        calcinfo.retrieve_list.append(
            self._OUTPUT_THERM_FILES_SUBFOLDER + self._OUTPUT_THERM_DEBYE_FILE
        )

        return calcinfo
