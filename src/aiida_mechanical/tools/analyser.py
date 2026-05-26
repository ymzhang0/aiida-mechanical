from aiida import orm
from aiida.engine import ProcessState
from enum import Enum
from collections import OrderedDict


class ThermoPwBaseWorkChainState(Enum):
    """
    Analyser for the ThermoPwBaseWorkChain.
    """

    FINISHED_OK = 0
    WAITING = 1
    RUNNING = 2
    EXCEPTED = 3
    KILLED = 4
    ERROR_TOO_MANY_PROCESSES = 5
    ERROR_TIME_LIMIT = 6
    ERROR_NSTEP = 7
    UNKNOWN = 999


class ThermoPwBaseAnalyser:
    """
    Analyser for the ThermoPwBaseWorkChain.
    """

    _all_descendants = OrderedDict(
        [
            ("thermo_pw", None),
        ]
    )

    def __init__(self, workchain: orm.WorkChainNode):
        self.node = workchain
        self.state = ThermoPwBaseWorkChainState.UNKNOWN
        self.descendants = {}
        for link_label, _ in self._all_descendants.items():
            descendants = workchain.base.links.get_outgoing(
                link_label_filter=link_label
            ).all_nodes()
            if descendants != []:
                self.descendants[link_label] = descendants

    def get_iterations(self):
        """Get the iterations of the workchain."""

        iterations = []
        for node, link_type, link_label in self.node.base.links.get_outgoing().all():
            if link_label.startswith("iteration"):
                iterations.append(node)
        return iterations

    def check_process_state(self):
        state = ThermoPwBaseWorkChainState.UNKNOWN
        message = ""

        source_db, source_id = (
            self.node.inputs.thermo_pw.structure.base.extras.get_many(
                ("source_db", "source_id")
            )
        )
        source = f"{source_db}-{source_id}"

        if self.node.process_state == ProcessState.WAITING:
            state = ThermoPwBaseWorkChainState.WAITING
            message = f"{self.node.process_state}"
        elif self.node.process_state == ProcessState.RUNNING:
            state = ThermoPwBaseWorkChainState.RUNNING
            message = f"{self.node.process_state}"
        elif self.node.process_state == ProcessState.EXCEPTED:
            state = ThermoPwBaseWorkChainState.EXCEPTED
            message = f"{self.node.process_state}"
        elif self.node.process_state == ProcessState.FINISHED:
            if self.node.exit_status != 0:
                final_iteration = self.get_iterations()[-1]
                if final_iteration.is_killed:
                    state = ThermoPwBaseWorkChainState.KILLED
                    message = f"{self.node.process_state} killed in final iteration [{final_iteration.pk}]"
                    return state, message
                if final_iteration.is_excepted:
                    state = ThermoPwBaseWorkChainState.EXCEPTED
                    message = f"{self.node.process_state} excepted in final iteration [{final_iteration.pk}]"
                    return state, message
                stderr = final_iteration.get_scheduler_stderr()
                aiida_out = final_iteration.outputs.retrieved.get_object_content(
                    "aiida.out"
                )
                for error, error_message in (
                    (
                        ThermoPwBaseWorkChainState.ERROR_TOO_MANY_PROCESSES,
                        "there are processes with no planes.",
                    ),
                    (ThermoPwBaseWorkChainState.ERROR_TIME_LIMIT, "TIME LIMIT"),
                ):
                    if error_message in stderr:
                        state = error
                        message = (
                            f"{error_message} in final iteration [{final_iteration.pk}]"
                        )
                        return state, message
                for error, error_message in (
                    (
                        ThermoPwBaseWorkChainState.ERROR_NSTEP,
                        "Incorrect nstep, check elastic_algorithm",
                    ),
                ):
                    if error_message in aiida_out:
                        state = error
                        message = (
                            f"{error_message} in final iteration [{final_iteration.pk}]"
                        )
                        return state, message
                state = ThermoPwBaseWorkChainState.FINISHED_OK
                message = f"{self.node.process_state} finished with exit status {self.node.exit_status} and final iteration [{final_iteration.pk}]"
                return state, message
            else:
                state = ThermoPwBaseWorkChainState.FINISHED_OK
                message = f"{self.node.process_state} finished with exit status {self.node.exit_status}"
                return state, message
        else:
            state = ThermoPwBaseWorkChainState.UNKNOWN
            message = f"{self.node.process_state} "
            return state, message

        return state, message

    def delete_nodes_and_remote_folder(
        self,
    ):
        from aiida.common import NotExistentAttributeError
        from aiida.tools import delete_nodes

        for called_descendant in self.node.called_descendants:
            if isinstance(called_descendant, orm.CalcJobNode):
                try:
                    called_descendant.outputs.remote_folder._clean()  # pylint: disable=protected-access
                except (IOError, OSError, KeyError, NotExistentAttributeError):
                    pass

        delete_nodes([self.node.pk], dry_run=False)
