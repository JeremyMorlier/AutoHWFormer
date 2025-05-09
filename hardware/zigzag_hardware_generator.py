
def zigzag_edge_tpu_hardware(xPEs, yPEs, n_SIMDS, n_compute_lanes, PE_memory, register_file_size) :
    core = {
    "name": "edge_tpu_like",
    "memories": {
        "rf_1B": {
            "size": 8,
            "r_bw": 8,
            "w_bw": 8,
            "r_cost": 0.01,
            "w_cost": 0.01,
            "area": 0,
            "r_port": 1,
            "w_port": 1,
            "rw_port": 0,
            "latency": 1,
            "auto_cost_extraction": False,
            "operands": ["I2"],
            "ports": [
                {
                    "fh": "w_port_1",
                    "tl": "r_port_1"
                }
            ],
            "served_dimensions": ["D3", "D4"]
        },
        "rf_2B": {
            "size": 16,
            "r_bw": 16,
            "w_bw": 16,
            "r_cost": 0.02,
            "w_cost": 0.02,
            "area": 0,
            "r_port": 2,
            "w_port": 2,
            "rw_port": 0,
            "latency": 1,
            "operands": ["O"],
            "ports": [
                {
                    "fh": "w_port_1",
                    "tl": "r_port_1",
                    "fl": "w_port_2",
                    "th": "r_port_2"
                }
            ],
            "served_dimensions": ["D3", "D4"]
        },
        "sram_32KB": {
            "size": register_file_size,
            "r_bw": 512,
            "w_bw": 512,
            "r_cost": 22.9,
            "w_cost": 52.01,
            "area": 0,
            "r_port": 1,
            "w_port": 1,
            "rw_port": 0,
            "latency": 1,
            "min_r_granularity": 64,
            "min_w_granularity": 64,
            "operands": ["I2"],
            "ports": [
                {
                    "fh": "w_port_1",
                    "tl": "r_port_1"
                }
            ],
            "served_dimensions": ["D3", "D4"]
        },
        "sram_2MB": {
            "size": PE_memory,
            "r_bw": 2048,
            "w_bw": 2048,
            "r_cost": 416.16,
            "w_cost": 378.4,
            "area": 0,
            "r_port": 1,
            "w_port": 1,
            "rw_port": 0,
            "latency": 1,
            "min_r_granularity": 64,
            "min_w_granularity": 64,
            "operands": ["I1", "I2", "O"],
            "ports": [
                {
                    "fh": "w_port_1",
                    "tl": "r_port_1"
                },
                {
                    "fh": "w_port_1",
                    "tl": "r_port_1"
                },
                {
                    "fh": "w_port_1",
                    "tl": "r_port_1",
                    "fl": "w_port_1",
                    "th": "r_port_1"
                }
            ],
            "served_dimensions": ["D3", "D4"]
        },
        "dram": {
            "size": 10000000000,
            "r_bw": 64,
            "w_bw": 64,
            "r_cost": 700,
            "w_cost": 750,
            "area": 0,
            "r_port": 0,
            "w_port": 0,
            "rw_port": 1,
            "latency": 1,
            "operands": ["I1", "I2", "O"],
            "ports": [
                {
                    "fh": "rw_port_1",
                    "tl": "rw_port_1"
                },
                {
                    "fh": "rw_port_1",
                    "tl": "rw_port_1"
                },
                {
                    "fh": "rw_port_1",
                    "tl": "rw_port_1",
                    "fl": "rw_port_1",
                    "th": "rw_port_1"
                }
            ],
            "served_dimensions": ["D1", "D2", "D3", "D4"]
        }
    },
    "operational_array": {
        "unit_energy": 0.04,  # pJ
        "unit_area": 1,  # unit
        "dimensions": ["D1", "D2", "D3", "D4"],
        "sizes": [xPEs, yPEs, n_SIMDS, n_compute_lanes]
    }
}
    return core

def zigzag_edge_tpu_mapping() :
    mapping =[
    {
        "name": "default",
        "spatial_mapping": {
            "D1": ["K, 8"],
            "D2": ["C, 8"],
            "D3": ["OX, 4"],
            "D4": ["OY, 4"]
        },
        "memory_operand_links": {
            "O": "O",
            "W": "I2",
            "I": "I1"
        }
    },
    {
        "name": "Add",
        "spatial_mapping": {
            "D1": ["G, 8"],
            "D2": ["C, 1"],
            "D3": ["OX, 1"],
            "D4": ["OY, 1"]
        },
        "memory_operand_links": {
            "O": "O",
            "W": "I2",
            "I": "I1"
        }
    },
    {
        "name": "Pooling",
        "spatial_mapping": {
            "D1": ["G, 8"],
            "D2": ["C, 1"],
            "D3": ["OX, 1"],
            "D4": ["OY, 1"]
        },
        "memory_operand_links": {
            "O": "O",
            "W": "I2",
            "I": "I1"
        }
    }
]

    return mapping