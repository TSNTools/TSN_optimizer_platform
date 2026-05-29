# TSN_optimizer_platform
A platform to develop optimization algorithms for TSN networks

# function description (used for selected and composed functions)
```json
"name"          : "exact function name as it appears in the context",
"type"          : "classmethod | staticmethod | instance_method | function | property",
"module"        : "source filename (e.g. \"usecase_generator_functions.md\")",
"class"         : "class the function belongs to, or null for module-level functions",
"description"   : "one sentence describing what the function does, inferred from its code/docstring",
"signature"     : "full def line verbatim from context (e.g. \"def k_ary_fat_tree(cls, k, directed=True):\")",
"parameters"    : [
                    {
                        "name": "parameter name",
                        "default": "default value or null if required"
                    }
                    ],
"preconditions" : ["conditions that must be true before calling the function"],
"postconditions": [
                    "guarantees about state changes after execution",
                    "includes return value semantics (e.g. what is returned and its structure)",
                    "includes side effects (e.g. mutations, I/O, logging, external updates)"
                    ],
"callable"        : true,
"dependencies"  :   {
                        "internal": [
                            "project-level functions, classes, or modules used by this function",
                            "may include calls like nx.shortest_path, np.zeros, etc."
                        ],
                        "external": [
                            "third-party libraries or external services not implemented in the codebase"
                        ]
                    },
"raises"       : ["exceptions explicitly raised or propagated by the function"],
"example_use"  : "short realistic example showing how the function is called",
"code"         : "complete function code verbatim from context",
"unit_tests"   : ["Tests functions covering normal cases, edge cases, and errors (inputs, outputs, exceptions)."]
```