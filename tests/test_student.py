"""Student protocol: scratch only, method choice, PyTorch only in GPU solves, and the sandbox fixes."""

import shutil
import unittest

from pde_rsi.code_agents import CodeVersion
from pde_rsi.codegen import ProgramRejected, check_program, run_program_task
from pde_rsi.identification import PROGRAM_KEY, StudentProtocol
from pde_rsi.runner import Budget
from pde_rsi.storage import PROJECT_ROOT
from pde_rsi.tasks import load_tasks

HAS_BWRAP = shutil.which("bwrap") is not None
from pathlib import Path


def has_gpu():
    try:
        from pde_rsi.codegen import gpu_env_prefix
        gpu_env_prefix()
    except (RuntimeError, OSError, ValueError):
        return False
    return any(Path("/dev").glob("nvidia[0-9]*"))


HAS_GPU = has_gpu()


class StudentProtocolTests(unittest.TestCase):
    def setUp(self):
        self.protocol = StudentProtocol()
        self.tasks = self.protocol.load_tasks(PROJECT_ROOT, None, ())

    def test_scratch_only_one_case_per_family(self):
        self.assertEqual({k: len(v) for k, v in self.tasks.items()}, {"evolution": 8, "development": 8})
        self.assertEqual(len({c.family for c in self.tasks["development"]}), 8)
        self.assertIn("NotImplementedError", self.protocol.starting_program("scratch", ())[PROGRAM_KEY])
        with self.assertRaises(ValueError):
            self.protocol.starting_program("classical", ())
        self.assertEqual(len(self.protocol.self_test_tasks(self.tasks["evolution"])), 3)
        self.assertEqual(self.tasks["development"][0].truth.max_seconds, 90.0)

    def test_contract_offers_numerical_and_pinn(self):
        contract = self.protocol.contract(())
        self.assertIn("pinn", contract)
        self.assertIn('model["method"]', contract)
        self.assertNotIn("DeepONet", contract)

    def test_torch_entry_points_that_load_or_compile_code_are_rejected(self):
        body = "import torch\ndef solve(problem):\n    return None\n"
        for line in ("torch.hub.load('x')", "torch.jit.script(f)", "torch.compile(f)", "torch.save(x, 'f')",
                     "torch.utils.cpp_extension.load('x')", "torch.multiprocessing.spawn(f)"):
            with self.subTest(line), self.assertRaises(ProgramRejected):
                check_program(body+line+"\n")
        self.assertTrue(check_program("import torch\nimport torch.nn as nn\ndef solve(problem):\n    return nn.Linear(1, 1)\n"))


@unittest.skipUnless(HAS_BWRAP, "bubblewrap is required for sandbox tests")
class SandboxFixTests(unittest.TestCase):
    task = [t for t in load_tasks(PROJECT_ROOT/"benchmarks/development.jsonl") if t.family == "poisson"][0]

    def test_matrix_products_work_in_the_sandbox(self):
        code = ("import numpy as np\ndef solve(problem):\n    a = np.ones((400, 400))\n    b = a @ a\n"
                "    x = np.linspace(0, problem.length, 65)\n    return x, np.array([0.0]), np.zeros((1, 65))*b[0, 0]\n")
        self.assertEqual(run_program_task(self.task, code, Budget(1, 60))["status"], "ok")

    def test_torch_is_not_importable_in_cpu_solves(self):
        code = "import torch\ndef solve(problem):\n    return None\n"
        result = run_program_task(self.task, code, Budget(1, 60))
        self.assertEqual(result["status"], "execution_failure")
        self.assertIn("not available", result["message"])


@unittest.skipUnless(HAS_BWRAP and HAS_GPU, "needs bubblewrap and a CUDA GPU")
class GpuSandboxTests(unittest.TestCase):
    def test_gpu_sandbox_is_isolated_and_has_cuda(self):
        from pde_rsi.codegen import gpu_sandbox_self_test
        report = gpu_sandbox_self_test()
        self.assertTrue(report["isolated"])
        self.assertTrue(report["cuda"])

    def test_tiny_pinn_runs_on_the_gpu(self):
        protocol = StudentProtocol()
        case = protocol.load_tasks(PROJECT_ROOT, None, ())["evolution"][0]
        code = '''import numpy as np
import torch
def solve(problem):
    dev = torch.device(problem.device)
    net = torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    pts = torch.tensor(np.column_stack([problem.data_x, problem.data_t]), dtype=torch.float32, device=dev)
    target = torch.tensor(problem.data_u, dtype=torch.float32, device=dev)
    for _ in range(50):
        opt.zero_grad()
        loss = ((net(pts)[:, 0]-target)**2).mean()
        loss.backward()
        opt.step()
    x = np.linspace(0, problem.length, 33)
    t = np.linspace(0, problem.final_time, 9)
    u = np.zeros((9, 33))
    u[0] = problem.initial(x)
    return {"x": x, "t": t, "u": u, "model": {"family": "heat", "parameters": {"D": 0.1}, "method": "pinn"}}
'''
        version = CodeVersion("m", program={PROGRAM_KEY: code})
        result = protocol.evaluate([case], version, Budget(1, 120, max_cpu_seconds=120))[0]
        self.assertEqual(result["status"], "ok", result.get("message"))
        self.assertEqual(result["method"], "pinn")


if __name__ == "__main__":
    unittest.main()
