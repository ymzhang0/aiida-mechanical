"""Parser for the thermo_pw.x calculation."""

import re
from aiida import orm
import numpy

from aiida_mechanical.calculations.thermo_pw import Thermo_pwCalculation

from aiida_quantumespresso.parsers.base import BaseParser

from aiida_quantumespresso.utils.mapping import get_logging_container


class Thermo_pwParser(BaseParser):
    """Parser for the thermo_pw.x calculation."""

    success_string = "JOB DONE."

    _PREFIX = Thermo_pwCalculation._PREFIX
    _OUTPUT_SUBFOLDER = Thermo_pwCalculation._OUTPUT_SUBFOLDER
    _OUTPUT_ELASTIC_CONSTANTS_SUBFOLDER = (
        Thermo_pwCalculation._OUTPUT_ELASTIC_CONSTANTS_SUBFOLDER
    )
    _OUTPUT_GNUPLOT_FILES_SUBFOLDER = (
        Thermo_pwCalculation._OUTPUT_GNUPLOT_FILES_SUBFOLDER
    )
    _OUTPUT_THERM_FILES_SUBFOLDER = Thermo_pwCalculation._OUTPUT_THERM_FILES_SUBFOLDER
    _OUTPUT_ELASTIC_CONSTANTS_FILE = Thermo_pwCalculation._OUTPUT_ELASTIC_CONSTANTS_FILE
    _OUTPUT_THERM_DEBYE_FILE = Thermo_pwCalculation._OUTPUT_THERM_DEBYE_FILE

    class_error_map = {
        "Error in routine latgen": "ERROR_LATGEN",
    }

    def parse(self, **kwargs):
        """Parse the retrieved files of a completed ``EpwCalculation`` into output nodes."""
        logs = get_logging_container()

        stdout, parsed_data, logs = self.parse_stdout_from_retrieved(logs)

        base_exit_code = self.check_base_errors(logs)
        if base_exit_code:
            return self.exit(base_exit_code, logs)

        # --- parse stdout ---
        parsed_thermo_pw, logs = self.parse_stdout(stdout, logs)
        parsed_data.update(parsed_thermo_pw)

        # --- elastic constants ---
        if (
            self._OUTPUT_ELASTIC_CONSTANTS_FILE
            in self.retrieved.base.repository.list_object_names()
        ):
            elastic_constants_contents = (
                self.retrieved.base.repository.get_object_content(
                    self._OUTPUT_ELASTIC_CONSTANTS_FILE
                )
            )
            self.out(
                "elastic_constants",
                self.parse_elastic_constants(elastic_constants_contents),
            )

        # --- thermal properties ---
        if (
            self._OUTPUT_THERM_DEBYE_FILE
            in self.retrieved.base.repository.list_object_names()
        ):
            therm_dat_debye_contents = (
                self.retrieved.base.repository.get_object_content(
                    self._OUTPUT_THERM_DEBYE_FILE
                )
            )
            self.out(
                "therm_dat_debye", self.parse_therm_dat_debye(therm_dat_debye_contents)
            )

        self.out("output_parameters", orm.Dict(parsed_data))

        for exit_code in list(self.get_error_map().values()):
            if exit_code in logs.error:
                return self.exit(self.exit_codes.get(exit_code), logs)

        # return self.exit(logs=logs)

        if "ERROR_OUTPUT_STDOUT_INCOMPLETE" in logs.error:
            return self.exit(
                self.exit_codes.get("ERROR_OUTPUT_STDOUT_INCOMPLETE"), logs
            )

        return self.exit(logs=logs)

    @staticmethod
    def parse_stdout(stdout, logs):
        """Parse the ``stdout``."""

        parsed_data = {}

        ANY_UNTIL_NEXT = r"[\s\S]*?"
        FLOAT_NUM = r"([\d\.]+)"

        data_type_regex = [
            # --- general information ---
            ("space_group_number", int, re.compile(r"Space group number\s+(\d+)")),
            ("space_group_symbol", str, re.compile(r"Space group\s+([Ff][m\-3m]+)")),
            ("laue_class", str, re.compile(r"The Laue class is\s+(.*)")),
            ("required_strains", str, re.compile(r"It requires .* strains:\s*(.*)")),
            (
                "total_scf_calculations",
                int,
                re.compile(r"for a total of\s+(\d+)\s+scf calculations"),
            ),
            # --- Voigt approximation ---
            # --- Reuss approximation ---
            # --- Voigt-Reuss-Hill average ---
            # --- Voigt-Reuss-Hill sound velocities ---
            # --- Debye temperatures and miscellaneous ---
            (
                "approx_debye_temp",
                float,
                re.compile(r"The approximate Debye temperature is\s*" + FLOAT_NUM),
            ),
            (
                "avg_debye_velocity",
                float,
                re.compile(r"Average Debye sound velocity\s*=\s*" + FLOAT_NUM),
            ),
            ("debye_temp", float, re.compile(r"Debye temperature\s*=\s*" + FLOAT_NUM)),
        ]
        stdout_lines = stdout.split("\n")

        for line_number, line in enumerate(stdout_lines):
            for data_key, data_type, re_pattern in data_type_regex:
                match = re.search(re_pattern, line)
                if match:
                    parsed_data[data_key] = data_type(match.group(1))

        # --- elastic constants type ---
        def parse_elastic_constants_type(stdout):
            single_row_pattern = r"\s*\(\s*(?:c\d{2}|\.)(?:\s+(?:c\d{2}|\.)){5}\s*\)"
            re_pattern = re.compile(
                r"In this class the elastic tensor is\s*\n\s*"
                r"("
                r"(?:" + single_row_pattern + r"[\r\n\s]*){6}"
                r")"
            )
            match = re_pattern.search(stdout)
            if match:
                elastic_tensor_raw = str(match.group(1))
                elastic_constants_type = [
                    line.split()[1:-1] for line in elastic_tensor_raw.split("\n")[:6]
                ]

                return elastic_constants_type
            else:
                return None

        elastic_constants_type = parse_elastic_constants_type(stdout)

        if elastic_constants_type:
            parsed_data["elastic_constants_type"] = elastic_constants_type

        # --- elastic constants fitting information ---
        def parse_elastic_constants_fitting(stdout):
            block_pattern = re.compile(
                r"Elastic constant\s+(\d+)\s+(\d+)\s*"
                r"strain\s+stress \(kbar\)\s*"
                r"((?:\s*-?\d+\.\d+\s+[-.\dE+]+\s*)+)"
                r"\s*Polynomial coefficients\s*"
                r"a1=\s*([-\d\.E+]+)\s*"
                r"a2=\s*([-\d\.E+]+)\s*"
                r"a3=\s*([-\d\.E+]+)"
            )

            elastic_constants_fitting = {}

            for match in block_pattern.finditer(stdout):
                i = int(match.group(1))
                j = int(match.group(2))
                table_raw_string = match.group(3)

                strains = []
                stresses = []
                strain_stress_pairs = re.findall(
                    r"(-?\d+\.\d+)\s+([-\d\.E+]+)", table_raw_string
                )
                for strain, stress in strain_stress_pairs:
                    strains.append(float(strain))
                    stresses.append(float(stress))

                coefficients = [
                    float(match.group(4)),
                    float(match.group(5)),
                    float(match.group(6)),
                ]

                if i not in elastic_constants_fitting:
                    elastic_constants_fitting[i] = {}

                elastic_constants_fitting[i][j] = {
                    "strains": strains,
                    "stresses": stresses,
                    "coefficients": coefficients,
                }

            return elastic_constants_fitting

        elastic_constants_fitting = parse_elastic_constants_fitting(stdout)
        parsed_data["elastic_constants_fitting"] = elastic_constants_fitting

        def parse_moduli(stdout):
            import re

            results = {"moduli": {}, "sound_velocities": {}}

            # Define all properties and their regex in one structure
            # CORRECTED: All value-capturing groups now use (-?[\d\.]+)
            definitions = {
                "voigt": {
                    "block_regex": re.compile(
                        r"Voigt approximation:\s*([\s\S]*?)(?=\n\s*Reuss approximation:|\Z)"
                    ),
                    "target_dict": results["moduli"],
                    "properties": {
                        "bulk_modulus_B": re.compile(
                            r"Bulk modulus\s+B\s*=\s*(-?[\d\.]+)"
                        ),
                        "young_modulus_E": re.compile(
                            r"Young modulus\s+E\s*=\s*(-?[\d\.]+)"
                        ),
                        "shear_modulus_G": re.compile(
                            r"Shear modulus\s+G\s*=\s*(-?[\d\.]+)"
                        ),
                        "poisson_ratio_n": re.compile(
                            r"Poisson Ratio\s+n\s*=\s*(-?[\d\.]+)"
                        ),
                        "pugh_ratio_r": re.compile(r"Pugh Ratio\s+r\s*=\s*(-?[\d\.]+)"),
                    },
                },
                "reuss": {
                    "block_regex": re.compile(
                        r"Reuss approximation:\s*([\s\S]*?)(?=\n\s*Voigt-Reuss-Hill average|\Z)"
                    ),
                    "target_dict": results["moduli"],
                    "properties": {
                        "bulk_modulus_B": re.compile(
                            r"Bulk modulus\s+B\s*=\s*(-?[\d\.]+)"
                        ),
                        "young_modulus_E": re.compile(
                            r"Young modulus\s+E\s*=\s*(-?[\d\.]+)"
                        ),
                        "shear_modulus_G": re.compile(
                            r"Shear modulus\s+G\s*=\s*(-?[\d\.]+)"
                        ),
                        "poisson_ratio_n": re.compile(
                            r"Poisson Ratio\s+n\s*=\s*(-?[\d\.]+)"
                        ),
                        "pugh_ratio_r": re.compile(r"Pugh Ratio\s+r\s*=\s*(-?[\d\.]+)"),
                    },
                },
                "vrh": {
                    "block_regex": re.compile(
                        r"Voigt-Reuss-Hill average of the two approximations:\s*([\s\S]*?)(?=\n\s*Voigt-Reuss-Hill average; sound velocities:|\Z)"
                    ),
                    "target_dict": results["moduli"],
                    "properties": {
                        "bulk_modulus_B": re.compile(
                            r"Bulk modulus\s+B\s*=\s*(-?[\d\.]+)"
                        ),
                        "young_modulus_E": re.compile(
                            r"Young modulus\s+E\s*=\s*(-?[\d\.]+)"
                        ),
                        "shear_modulus_G": re.compile(
                            r"Shear modulus\s+G\s*=\s*(-?[\d\.]+)"
                        ),
                        "longitudinal_modulus_L": re.compile(
                            r"Longitudinal modulus\s+L\s*=\s*(-?[\d\.]+)"
                        ),
                        "poisson_ratio_n": re.compile(
                            r"Poisson Ratio\s+n\s*=\s*(-?[\d\.]+)"
                        ),
                        "pugh_ratio_r": re.compile(r"Pugh Ratio\s+r\s*=\s*(-?[\d\.]+)"),
                    },
                },
                "velocities": {
                    "block_regex": re.compile(
                        r"Voigt-Reuss-Hill average; sound velocities:\s*([\s\S]*?)(?=\n\s*\n|\Z)"
                    ),
                    "target_dict": results,  # The target is the top-level results for velocities
                    "target_key": "sound_velocities",  # The key within the target dict
                    "properties": {
                        "compressional_V_P": re.compile(
                            r"Compressional\s+V_P\s*=\s*(-?[\d\.]+)"
                        ),
                        "bulk_V_B": re.compile(r"Bulk\s+V_B\s*=\s*(-?[\d\.]+)"),
                        "shear_V_G": re.compile(r"Shear\s+V_G\s*=\s*(-?[\d\.]+)"),
                    },
                },
            }

            # A single loop to process everything
            for block_name, definition in definitions.items():
                block_match = definition["block_regex"].search(stdout)
                if not block_match:
                    continue

                block_text = block_match.group(1)

                # Determine where to store the results for this block
                if "target_key" in definition:
                    # For sound_velocities, which is a direct key in results
                    storage = definition["target_dict"][definition["target_key"]]
                else:
                    # For moduli, which are nested under their block name (voigt, reuss, etc.)
                    definition["target_dict"][block_name] = {}
                    storage = definition["target_dict"][block_name]

                # Parse properties within the block
                for prop_name, prop_regex in definition["properties"].items():
                    prop_match = prop_regex.search(block_text)
                    if prop_match:
                        storage[prop_name] = float(prop_match.group(1))

            return results

        moduli = parse_moduli(stdout)
        parsed_data.update(moduli)

        return parsed_data, logs

    @staticmethod
    def parse_elastic_constants(content):
        elastic_constants_array = orm.ArrayData()

        matrix_blocks = content.strip().split("\n \n")

        elastic_constants = numpy.array(matrix_blocks[0].split(), dtype=float).reshape(
            (6, 6)
        )
        elastic_compliances = numpy.array(
            matrix_blocks[1].split(), dtype=float
        ).reshape((6, 6))

        elastic_constants_array.set_array("elastic_constants", elastic_constants)
        elastic_constants_array.set_array("elastic_compliances", elastic_compliances)

        return elastic_constants_array

    @staticmethod
    def parse_therm_dat_debye(content):
        import io

        therm_dat_debye_xydata = orm.XyData()
        therm_dat_debye = numpy.loadtxt(
            io.StringIO((content)), dtype=float, comments="#"
        )
        therm_dat_debye_xydata.set_array("Temperature", therm_dat_debye[:, 0])
        therm_dat_debye_xydata.set_array("Energy", therm_dat_debye[:, 1])
        therm_dat_debye_xydata.set_array("Free_energy", therm_dat_debye[:, 2])
        therm_dat_debye_xydata.set_array("Entropy", therm_dat_debye[:, 3])
        therm_dat_debye_xydata.set_array("Cv", therm_dat_debye[:, 4])
        return therm_dat_debye_xydata
