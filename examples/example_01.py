#!/usr/bin/env python
"""Run a ``Thermo_pwBaseWorkChain`` for a scf calculation.

Usage: ./example_01.py
"""

import click

from aiida import cmdline, orm

from aiida_mechanical.workflows.base import Thermo_pwBaseWorkChain

from aiida_mechanical.cli.params import RUN
from aiida_mechanical.utils.structure import read_structure
from aiida_mechanical.utils.workflows.builder.serializer import print_builder
from aiida_mechanical.utils.workflows.builder.setter import set_parallelization
from aiida_mechanical.utils.workflows.builder.submit import (
    submit_and_add_group,
)


def submit(
    code: orm.Code,
    structure: orm.StructureData,
    group: orm.Group = None,
    run: bool = False,
):
    """Submit a ``Thermo_pwBaseWorkChain`` to calculate the thermodynamic properties."""
    builder = Thermo_pwBaseWorkChain.get_builder_from_protocol(
        code, structure=structure
    )

    # You can change parallelization here
    parallelization = {
        "num_mpiprocs_per_machine": 8,
        "npool": 4,
    }
    set_parallelization(builder, parallelization, process_class=Thermo_pwBaseWorkChain)

    print_builder(builder)

    if run:
        submit_and_add_group(builder, group)


@click.command()
@cmdline.utils.decorators.with_dbenv()
@cmdline.params.options.CODE(help="The pw.x code identified by its ID, UUID or label.")
@cmdline.params.options.GROUP(help="The group to add the submitted workchain.")
@click.argument("filename", type=click.Path(exists=True))
@RUN()
def cli(filename, code, group, run):
    """Run a ``Thermo_pwBaseWorkChain`` to calculate the scf.

    FILENAME: a crystal structure file, e.g., ``structures/Si.xsf``.
    """
    struct = read_structure(filename, store=True)
    submit(code, struct, group, run)
