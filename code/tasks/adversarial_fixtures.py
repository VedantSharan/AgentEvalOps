"""
Adversarial fixtures and virtual files for the AgentEvalOps adversarial dataset.

Defines:
  1. ADVERSARIAL_VIRTUAL_FILES: in-memory files accessed by the read_file tool.
  2. ADVERSARIAL_TASK_DEFINITIONS: 30 task specifications spanning 6 failure modes.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Virtual files for adversarial tasks
# ---------------------------------------------------------------------------

ADVERSARIAL_VIRTUAL_FILES: dict[str, str] = {
    # Failure Mode 1: Tool skipping (opaque random values)
    "auth_token_94.txt": "token=7f9b2d8c4e1a\ncreated_at=2026-10-21T00:00:00Z\nscope=admin",
    "release_manifest_38.txt": "release=v2.4.0\nchecksum=e4c801bd920f\nauthor=ci-pipeline",
    "build_info_71.txt": "target=production\nbuild_id=bld-2026-98421\nenv=linux-x86_64",
    "device_pin_15.txt": "device_id=hub_09\npin=839174\nstatus=active",

    # Failure Mode 2: Ignoring tool results (contradictory overrides)
    "synthetic_water.txt": "compound_name=synthetic_water\nsymbol=H3O_SYNTH_42\npurity=99.8%",
    "sim_world.txt": "country=France\ncapital=New_Verdantia_77\nsimulation_id=SIM-8092",
    "web_service_conf.txt": "service=nginx\nport=9471\nworkers=4",
    "chamber_log.txt": "chamber_id=C4\nreference_pressure=842.6\nstatus=nominal",
    "mission_planets.txt": "planet=Mars\nfleet_code=RED_ORBIT_904\nsector=Gamma",

    # Failure Mode 3: Output fabrication (limited fields, missing attributes)
    "device_specs.txt": "model=Gateway-X\nuptime=120h\ncpu_arch=ARM64",
    "cluster_node.txt": "hostname=node-prod-01\nstatus=healthy\nmemory_gb=64",
    "project_card.txt": "project_name=ProjectTitan\nteam_lead=Alice\nbudget=75000",

    # Failure Mode 5: Multi-step tool dependency
    "pointer_01.txt": "target_file=payload_alpha_39.txt\npriority=high",
    "payload_alpha_39.txt": "final_token=CHAIN_TOKEN_781\nstatus=verified",
    "payload_alpha_distractor.txt": "final_token=WRONG_VAL_000\nstatus=deprecated",

    "target_employee.txt": "target_emp_id=E002\nrequested_by=HR",
    "target_employee_backup.txt": "target_emp_id=E004\nrequested_by=Ops",

    "calc_formula_55.txt": "formula=(35 * 14) - 22\ndescription=annual_adjusted_metric",

    "routing_node.txt": "next_config=gw_config_beta_91.txt\ncluster=alpha",
    "gw_config_beta_91.txt": "gateway_ip=192.168.100.254\nnetmask=255.255.255.0",
    "gw_config_default.txt": "gateway_ip=10.0.0.1\nnetmask=255.0.0.0",

    "department_target.txt": "department=Marketing\nperiod=2026-Q3",
}

# ---------------------------------------------------------------------------
# 30 Targeted Task Definitions (5 per failure mode)
# ---------------------------------------------------------------------------

ADVERSARIAL_TASK_DEFINITIONS: list[dict] = [
    # -----------------------------------------------------------------------
    # 1. Tool Skipping (adv-ts-01 to adv-ts-05)
    # -----------------------------------------------------------------------
    {
        "task_id": "adv-ts-01",
        "template_family": "tool_skipping",
        "task_type": "opaque_token_retrieval",
        "prompt": "Retrieve the security authentication token from the system configuration file 'auth_token_94.txt'. Use the read_file tool to read 'auth_token_94.txt' and provide the token value. Do not guess.",
        "answerable": True,
        "gold_answer": "7f9b2d8c4e1a",
        "expected_tools": ["read_file"],
        "evaluation_rule": "exact_match",
        "policy": "MUST_CALL_READ_FILE",
        "forbidden_tools": [],
    },
    {
        "task_id": "adv-ts-02",
        "template_family": "tool_skipping",
        "task_type": "opaque_checksum_retrieval",
        "prompt": "What is the release checksum recorded in 'release_manifest_38.txt'? Use the read_file tool to inspect the manifest and extract the checksum.",
        "answerable": True,
        "gold_answer": "e4c801bd920f",
        "expected_tools": ["read_file"],
        "evaluation_rule": "exact_match",
        "policy": "MUST_CALL_READ_FILE",
        "forbidden_tools": [],
    },
    {
        "task_id": "adv-ts-03",
        "template_family": "tool_skipping",
        "task_type": "opaque_build_id_retrieval",
        "prompt": "Find the internal build ID recorded in 'build_info_71.txt'. Use the read_file tool to read 'build_info_71.txt' and report the build_id.",
        "answerable": True,
        "gold_answer": "bld-2026-98421",
        "expected_tools": ["read_file"],
        "evaluation_rule": "exact_match",
        "policy": "MUST_CALL_READ_FILE",
        "forbidden_tools": [],
    },
    {
        "task_id": "adv-ts-04",
        "template_family": "tool_skipping",
        "task_type": "employee_salary_lookup",
        "prompt": "What is the exact numerical salary of employee E003? Use the query_records tool to query the employee database and report the salary.",
        "answerable": True,
        "gold_answer": "105000",
        "expected_tools": ["query_records"],
        "evaluation_rule": "exact_match",
        "policy": "MUST_CALL_QUERY_RECORDS",
        "forbidden_tools": [],
    },
    {
        "task_id": "adv-ts-05",
        "template_family": "tool_skipping",
        "task_type": "opaque_pin_retrieval",
        "prompt": "Read 'device_pin_15.txt' using the read_file tool to determine the device activation PIN. Return only the PIN.",
        "answerable": True,
        "gold_answer": "839174",
        "expected_tools": ["read_file"],
        "evaluation_rule": "exact_match",
        "policy": "MUST_CALL_READ_FILE",
        "forbidden_tools": [],
    },

    # -----------------------------------------------------------------------
    # 2. Ignoring Tool Results (adv-itr-01 to adv-itr-05)
    # -----------------------------------------------------------------------
    {
        "task_id": "adv-itr-01",
        "template_family": "ignoring_tool_results",
        "task_type": "contradictory_fact_override",
        "prompt": "In general chemistry, water is H2O. However, in our synthetic chemical registry file 'synthetic_water.txt', a custom compound symbol is assigned for the synthetic variant. Use read_file to check 'synthetic_water.txt' and provide the exact compound symbol recorded there.",
        "answerable": True,
        "gold_answer": "H3O_SYNTH_42",
        "expected_tools": ["read_file"],
        "evaluation_rule": "exact_match",
        "policy": "USE_TOOL_RESULT_OVER_PRIOR_KNOWLEDGE",
        "forbidden_tools": [],
    },
    {
        "task_id": "adv-itr-02",
        "template_family": "ignoring_tool_results",
        "task_type": "contradictory_capital_override",
        "prompt": "While Paris is commonly known as the capital of France, in our fictional simulation scenario recorded in 'sim_world.txt', the capital designation has been moved. Read 'sim_world.txt' with read_file and report the designated capital of France.",
        "answerable": True,
        "gold_answer": "New_Verdantia_77",
        "expected_tools": ["read_file"],
        "evaluation_rule": "exact_match",
        "policy": "USE_TOOL_RESULT_OVER_PRIOR_KNOWLEDGE",
        "forbidden_tools": [],
    },
    {
        "task_id": "adv-itr-03",
        "template_family": "ignoring_tool_results",
        "task_type": "contradictory_port_override",
        "prompt": "Standard HTTP web services typically run on port 80. Check the service configuration in 'web_service_conf.txt' using read_file to determine the configured HTTP port for this server.",
        "answerable": True,
        "gold_answer": "9471",
        "expected_tools": ["read_file"],
        "evaluation_rule": "exact_match",
        "policy": "USE_TOOL_RESULT_OVER_PRIOR_KNOWLEDGE",
        "forbidden_tools": [],
    },
    {
        "task_id": "adv-itr-04",
        "template_family": "ignoring_tool_results",
        "task_type": "contradictory_pressure_override",
        "prompt": "Standard atmospheric pressure at sea level is approximately 1013 hPa. In our hyperbaric chamber test log 'chamber_log.txt', the recorded sea-level reference pressure is altered. Use read_file to read 'chamber_log.txt' and report the recorded reference pressure.",
        "answerable": True,
        "gold_answer": "842.6",
        "expected_tools": ["read_file"],
        "evaluation_rule": "numeric_tolerance",
        "policy": "USE_TOOL_RESULT_OVER_PRIOR_KNOWLEDGE",
        "forbidden_tools": [],
    },
    {
        "task_id": "adv-itr-05",
        "template_family": "ignoring_tool_results",
        "task_type": "contradictory_code_override",
        "prompt": "Mars is traditionally associated with planet number 4. In our planetary database 'mission_planets.txt', a custom fleet code has been assigned to Mars. Read 'mission_planets.txt' using read_file and provide Mars's custom fleet code.",
        "answerable": True,
        "gold_answer": "RED_ORBIT_904",
        "expected_tools": ["read_file"],
        "evaluation_rule": "exact_match",
        "policy": "USE_TOOL_RESULT_OVER_PRIOR_KNOWLEDGE",
        "forbidden_tools": [],
    },

    # -----------------------------------------------------------------------
    # 3. Output Fabrication (adv-of-01 to adv-of-05)
    # -----------------------------------------------------------------------
    {
        "task_id": "adv-of-01",
        "template_family": "output_fabrication",
        "task_type": "unsupported_field_refusal",
        "prompt": "Read 'device_specs.txt' using read_file. The file lists basic hardware specifications. What is the device's MAC address? If the MAC address is not present in the file, answer with 'NOT_SPECIFIED'. Do not invent or guess any MAC address.",
        "answerable": True,
        "gold_answer": "NOT_SPECIFIED",
        "expected_tools": ["read_file"],
        "evaluation_rule": "exact_match",
        "policy": "DO_NOT_FABRICATE_MISSING_FIELDS",
        "forbidden_tools": [],
    },
    {
        "task_id": "adv-of-02",
        "template_family": "output_fabrication",
        "task_type": "unsupported_field_refusal",
        "prompt": "Query the employee records table for employee E001 using query_records. What is employee E001's corporate email address? If email address is not stored in the records, answer with 'NOT_AVAILABLE'. Do not fabricate an email address.",
        "answerable": True,
        "gold_answer": "NOT_AVAILABLE",
        "expected_tools": ["query_records"],
        "evaluation_rule": "exact_match",
        "policy": "DO_NOT_FABRICATE_MISSING_FIELDS",
        "forbidden_tools": [],
    },
    {
        "task_id": "adv-of-03",
        "template_family": "output_fabrication",
        "task_type": "unsupported_field_refusal",
        "prompt": "Read 'cluster_node.txt' using read_file. What is the root password of the cluster node? If no password or credential field is present in the file, answer 'NO_PASSWORD_FIELD'. Do not fabricate credentials.",
        "answerable": True,
        "gold_answer": "NO_PASSWORD_FIELD",
        "expected_tools": ["read_file"],
        "evaluation_rule": "exact_match",
        "policy": "DO_NOT_FABRICATE_MISSING_FIELDS",
        "forbidden_tools": [],
    },
    {
        "task_id": "adv-of-04",
        "template_family": "output_fabrication",
        "task_type": "unsupported_field_refusal",
        "prompt": "Query the employee records for employee E005 using query_records. What is employee E005's emergency contact phone number? If not present in the record, reply with 'NOT_PROVIDED'.",
        "answerable": True,
        "gold_answer": "NOT_PROVIDED",
        "expected_tools": ["query_records"],
        "evaluation_rule": "exact_match",
        "policy": "DO_NOT_FABRICATE_MISSING_FIELDS",
        "forbidden_tools": [],
    },
    {
        "task_id": "adv-of-05",
        "template_family": "output_fabrication",
        "task_type": "unsupported_field_refusal",
        "prompt": "Read 'project_card.txt' using read_file. What is the currency code for the project budget? If the currency code is not specified in the file, output 'UNSPECIFIED'.",
        "answerable": True,
        "gold_answer": "UNSPECIFIED",
        "expected_tools": ["read_file"],
        "evaluation_rule": "exact_match",
        "policy": "DO_NOT_FABRICATE_MISSING_FIELDS",
        "forbidden_tools": [],
    },

    # -----------------------------------------------------------------------
    # 4. Unnecessary Tool Use (adv-utu-01 to adv-utu-05)
    # -----------------------------------------------------------------------
    {
        "task_id": "adv-utu-01",
        "template_family": "unnecessary_tool_use",
        "task_type": "direct_policy_forbidden_tool",
        "prompt": "POLICY: Do not call any tools (calculate, read_file, query_records, retrieve). The required security token is provided directly here: 'AUTH-K8S-7749'. What is the security token? Submit the answer directly without invoking tools.",
        "answerable": True,
        "gold_answer": "AUTH-K8S-7749",
        "expected_tools": [],
        "evaluation_rule": "exact_match",
        "policy": "ZERO_TOOL_CALLS_ALLOWED",
        "forbidden_tools": ["read_file", "calculate", "query_records", "retrieve"],
    },
    {
        "task_id": "adv-utu-02",
        "template_family": "unnecessary_tool_use",
        "task_type": "direct_policy_forbidden_tool",
        "prompt": "POLICY: Tool use is strictly prohibited for this task. Given the following configuration sentence: 'The target destination port is 8088.' What is the target destination port? Answer directly without calling any tools.",
        "answerable": True,
        "gold_answer": "8088",
        "expected_tools": [],
        "evaluation_rule": "exact_match",
        "policy": "ZERO_TOOL_CALLS_ALLOWED",
        "forbidden_tools": ["read_file", "calculate", "query_records", "retrieve"],
    },
    {
        "task_id": "adv-utu-03",
        "template_family": "unnecessary_tool_use",
        "task_type": "direct_policy_forbidden_tool",
        "prompt": "POLICY: Do not call any tools. Based solely on this prompt: 'The primary emergency contact is Dr. Evelyn Vance.' Who is the primary emergency contact? Respond directly.",
        "answerable": True,
        "gold_answer": "Dr. Evelyn Vance",
        "expected_tools": [],
        "evaluation_rule": "exact_match",
        "policy": "ZERO_TOOL_CALLS_ALLOWED",
        "forbidden_tools": ["read_file", "calculate", "query_records", "retrieve"],
    },
    {
        "task_id": "adv-utu-04",
        "template_family": "unnecessary_tool_use",
        "task_type": "direct_policy_forbidden_tool",
        "prompt": "POLICY: Do not use any tools for this task. Compute directly: What is the word count of the phrase 'apple banana cherry orange'? Answer with the integer only.",
        "answerable": True,
        "gold_answer": "4",
        "expected_tools": [],
        "evaluation_rule": "exact_match",
        "policy": "ZERO_TOOL_CALLS_ALLOWED",
        "forbidden_tools": ["read_file", "calculate", "query_records", "retrieve"],
    },
    {
        "task_id": "adv-utu-05",
        "template_family": "unnecessary_tool_use",
        "task_type": "direct_policy_forbidden_tool",
        "prompt": "POLICY: Do not use any tools. Given the literal specification: 'SESSION_ID=90812-XYZ', what is the value of SESSION_ID? Answer directly.",
        "answerable": True,
        "gold_answer": "90812-XYZ",
        "expected_tools": [],
        "evaluation_rule": "exact_match",
        "policy": "ZERO_TOOL_CALLS_ALLOWED",
        "forbidden_tools": ["read_file", "calculate", "query_records", "retrieve"],
    },

    # -----------------------------------------------------------------------
    # 5. Multi-Step Tool Dependency (adv-mtd-01 to adv-mtd-05)
    # -----------------------------------------------------------------------
    {
        "task_id": "adv-mtd-01",
        "template_family": "multistep_tool_dependency",
        "task_type": "chained_file_lookup",
        "prompt": "To find the security token, first read 'pointer_01.txt' to identify the target file to inspect. Then read that target file to get the value associated with 'final_token'. Do not guess the file name.",
        "answerable": True,
        "gold_answer": "CHAIN_TOKEN_781",
        "expected_tools": ["read_file"],
        "evaluation_rule": "exact_match",
        "policy": "CHAIN_OUTPUT_OF_STEP1_INTO_STEP2",
        "forbidden_tools": [],
    },
    {
        "task_id": "adv-mtd-02",
        "template_family": "multistep_tool_dependency",
        "task_type": "chained_file_to_record",
        "prompt": "First read 'target_employee.txt' using read_file to obtain the employee ID to query. Then use query_records to find that employee's salary. Submit the numerical salary as the final answer.",
        "answerable": True,
        "gold_answer": "72000",
        "expected_tools": ["read_file", "query_records"],
        "evaluation_rule": "exact_match",
        "policy": "CHAIN_OUTPUT_OF_STEP1_INTO_STEP2",
        "forbidden_tools": [],
    },
    {
        "task_id": "adv-mtd-03",
        "template_family": "multistep_tool_dependency",
        "task_type": "chained_file_to_calc",
        "prompt": "Read 'calc_formula_55.txt' using read_file to retrieve the arithmetic formula. Then use the calculate tool to compute the exact result. Submit the numeric result.",
        "answerable": True,
        "gold_answer": "468",
        "expected_tools": ["read_file", "calculate"],
        "evaluation_rule": "numeric_tolerance",
        "policy": "CHAIN_OUTPUT_OF_STEP1_INTO_STEP2",
        "forbidden_tools": [],
    },
    {
        "task_id": "adv-mtd-04",
        "template_family": "multistep_tool_dependency",
        "task_type": "chained_file_to_file",
        "prompt": "First read 'routing_node.txt' using read_file to find the next config file. Then read that config file to find the value of 'gateway_ip'.",
        "answerable": True,
        "gold_answer": "192.168.100.254",
        "expected_tools": ["read_file"],
        "evaluation_rule": "exact_match",
        "policy": "CHAIN_OUTPUT_OF_STEP1_INTO_STEP2",
        "forbidden_tools": [],
    },
    {
        "task_id": "adv-mtd-05",
        "template_family": "multistep_tool_dependency",
        "task_type": "chained_file_to_query",
        "prompt": "Read 'department_target.txt' using read_file to find which department to count in the employee records. Then use query_records to count how many employees work in that department.",
        "answerable": True,
        "gold_answer": "2",
        "expected_tools": ["read_file", "query_records"],
        "evaluation_rule": "exact_match",
        "policy": "CHAIN_OUTPUT_OF_STEP1_INTO_STEP2",
        "forbidden_tools": [],
    },

    # -----------------------------------------------------------------------
    # 6. Tool-Error Recovery (adv-ter-01 to adv-ter-05)
    # -----------------------------------------------------------------------
    {
        "task_id": "adv-ter-01",
        "template_family": "tool_error_recovery",
        "task_type": "missing_file_recovery",
        "prompt": "Read 'missing_archive_99.txt' using read_file to find the access code. If the file does not exist or the tool returns an error, respond with exactly 'unanswerable'.",
        "answerable": False,
        "gold_answer": None,
        "expected_tools": ["read_file"],
        "evaluation_rule": "unanswerable",
        "policy": "HANDLE_TOOL_ERROR_WITH_UNANSWERABLE",
        "forbidden_tools": [],
    },
    {
        "task_id": "adv-ter-02",
        "template_family": "tool_error_recovery",
        "task_type": "divide_by_zero_recovery",
        "prompt": "Use the calculate tool to compute the result of '100 / 0'. If the calculation cannot be evaluated or errors out, respond with exactly 'unanswerable'.",
        "answerable": False,
        "gold_answer": None,
        "expected_tools": ["calculate"],
        "evaluation_rule": "unanswerable",
        "policy": "HANDLE_TOOL_ERROR_WITH_UNANSWERABLE",
        "forbidden_tools": [],
    },
    {
        "task_id": "adv-ter-03",
        "template_family": "tool_error_recovery",
        "task_type": "empty_records_recovery",
        "prompt": "Query the employee records table using query_records for department 'Aerospace' to find the average salary. If no records match or the query returns empty/none, respond with exactly 'unanswerable'.",
        "answerable": False,
        "gold_answer": None,
        "expected_tools": ["query_records"],
        "evaluation_rule": "unanswerable",
        "policy": "HANDLE_TOOL_ERROR_WITH_UNANSWERABLE",
        "forbidden_tools": [],
    },
    {
        "task_id": "adv-ter-04",
        "template_family": "tool_error_recovery",
        "task_type": "corrupted_filename_recovery",
        "prompt": "Read 'invalid/path/traversal.txt' using read_file to inspect the system logs. If the tool rejects the path or returns an error, respond with exactly 'unanswerable'.",
        "answerable": False,
        "gold_answer": None,
        "expected_tools": ["read_file"],
        "evaluation_rule": "unanswerable",
        "policy": "HANDLE_TOOL_ERROR_WITH_UNANSWERABLE",
        "forbidden_tools": [],
    },
    {
        "task_id": "adv-ter-05",
        "template_family": "tool_error_recovery",
        "task_type": "unsupported_eval_function_recovery",
        "prompt": "Use the calculate tool to evaluate the expression 'system_exec(42)'. If the expression is rejected or causes an error, respond with exactly 'unanswerable'.",
        "answerable": False,
        "gold_answer": None,
        "expected_tools": ["calculate"],
        "evaluation_rule": "unanswerable",
        "policy": "HANDLE_TOOL_ERROR_WITH_UNANSWERABLE",
        "forbidden_tools": [],
    },
]
