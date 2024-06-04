import subprocess

PROBLEMS = [
    "kinase",
    # "ampc",
    # "d4"
]
METHODS = [
    "gp",
    # "la",
]
WITH_EXPERT = [
    True,
    # False,
]
EXPERT_PROBS = [
    "0.1",
    # "0.25",
]

# Run sbatch
for method in METHODS:
    for problem in PROBLEMS:
        for with_expert in WITH_EXPERT:
            if with_expert:
                for expert_prob in EXPERT_PROBS:
                    subprocess.run(
                        [
                            "sh",
                            "scripts/run_array.sh",
                            problem,
                            method,
                            "--with-expert",
                            expert_prob,
                        ],
                    )
            else:
                subprocess.run(
                    [
                        "sh",
                        "scripts/run_array.sh",
                        problem,
                        method,
                        "",
                        "0",
                    ],
                )
