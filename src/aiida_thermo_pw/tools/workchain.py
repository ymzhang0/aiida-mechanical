from aiida import orm
from aiida.common.links import LinkType
from aiida.common import NotExistentAttributeError
from aiida.tools import delete_nodes
from rich import print as rprint
from aiida.common.links import LinkType
from aiida.engine import ProcessState

def get_descendants_by_label(
    parent_workchain: orm.WorkChainNode,
    link_label_filter: str
    ) -> orm.WorkChainNode:
    """Get the descendant workchains of the parent workchain by the link label."""
    try:
        return parent_workchain.base.links.get_outgoing(
            link_label_filter=link_label_filter
            ).all()
    except AttributeError:
        return None

def get_descendants_by_type(
    parent_workchain: orm.WorkChainNode,
    link_type: LinkType = LinkType.CALL_WORK
    ) -> dict:
    """Get the descendant nodes of the parent workchain."""

    descendants = {}
    try:
        for node, link_type, link_label in parent_workchain.base.links.get_outgoing(link_type=link_type).all():
            if link_label not in descendants:
                descendants[link_label] = []
            descendants[link_label].append(node)
        return descendants
    except AttributeError:
        return None

def get_workdirs(
    workchain: orm.WorkChainNode,
):
    workdirs = {}
    for node, link_type, link_label in workchain.base.links.get_outgoing(link_type=LinkType.CALL_CALC).all():
        if link_label.startswith('iteration_'): # only get the workdirs of the iterations
            workdirs[link_label] = node.get_remote_workdir()
    return workdirs

def get_iterations(
    workchain: orm.WorkChainNode,
):
    iterations = []
    for node, link_type, link_label in workchain.base.links.get_outgoing(link_type=LinkType.CALL_CALC).all():
        if link_label.startswith('iteration_'): # only get the workdirs of the iterations
            iterations.append(node)
    return iterations

def delete_nodes_and_remote_folder(
    process: orm.ProcessNode,
):
    for called_descendant in process.called_descendants:
        if isinstance(called_descendant, orm.CalcJobNode):
            try:
                called_descendant.outputs.remote_folder._clean()  # pylint: disable=protected-access
            except (IOError, OSError, KeyError, NotExistentAttributeError):
                pass

    delete_nodes([process.pk], dry_run=False)


def check_process_state(
    process: orm.ProcessNode,
    only_report_error: bool = True,
    ):
    from collections import deque
    from io import StringIO

    source_db, source_id = process.inputs.thermo_pw.structure.base.extras.get_many(('source_db', 'source_id'))
    source = f"{source_db}-{source_id}"

    if process.process_state == ProcessState.WAITING:
        rprint(f"[bold green] {source}[{process.pk}]: {process.process_state}")
    elif process.process_state == ProcessState.RUNNING:
        rprint(f"[bold green] {source}[{process.pk}]: {process.process_state}")
    elif process.process_state == ProcessState.EXCEPTED:
        rprint(f"[bold red] {source}[{process.pk}]: {process.process_state} excepted")
    elif process.process_state == ProcessState.FINISHED:
        if process.exit_status != 0:
            final_iteration = get_iterations(process)[-1]
            if final_iteration.is_killed:
                rprint(f"[bold red] {source}[{process.pk}]: {process.process_state} killed in final iteration [{final_iteration.pk}]")
                return source
            if final_iteration.is_excepted:
                rprint(f"[bold red] {source}[{process.pk}]: {process.process_state} excepted in final iteration [{final_iteration.pk}]")
                return source
            stderr = final_iteration.get_scheduler_stderr()
            aiida_out = final_iteration.outputs.retrieved.get_object_content("aiida.out")
            for error_flag, error_message in (
                ('ERROR_TOO_MANY_PROCESSES', 'there are processes with no planes.'),
                ('ERROR_TIME_LIMIT',  'TIME LIMIT'),
            ):
                if error_message in stderr:
                    rprint(f"[bold red] {source}[{process.pk}]: {error_flag} {error_message} in final iteration [{final_iteration.pk}]")
                    return source
            for error_flag, error_message in (
                ('ERROR_NSTEP', 'Incorrect nstep, check elastic_algorithm'),
             ):
                if error_message in aiida_out:
                    rprint(f"[bold red] {source}[{process.pk}]: {error_flag} {error_message} in final iteration [{final_iteration.pk}]")
                    return source
            rprint(f"[bold red] {source}[{process.pk}]: {process.process_state} finished with exit status {process.exit_status} and final iteration [{final_iteration.pk}]")
        elif not only_report_error:
            rprint(f"[bold green] {source}[{process.pk}]: {process.process_state} finished with exit status {process.exit_status}")
    else:
        rprint(f"[bold red] {source}[{process.pk}]: {process.process_state} ")

    return source