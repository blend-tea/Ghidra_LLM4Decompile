# Decompile current function using LLM4decompile
# @category LLM4Decompile
# @runtime PyGhidra
# @keybinding F5
# @author blend-tea


# LLM Requirements
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Ghidra Requirements
import typing
if typing.TYPE_CHECKING:
    from ghidra.ghidra_builtins import *

from ghidra.app.decompiler import DecompInterface
from ghidra.util.task import ConsoleTaskMonitor

def load_model():
    model_path = 'LLM4Binary/llm4decompile-1.3b-v2' # V2 Model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).cuda()
    return tokenizer, model

def decompile_function(function):
    decompiler_interface = DecompInterface()
    decompiler_interface.openProgram(currentProgram)
    decompiled_function = decompiler_interface.decompileFunction(function, 0, ConsoleTaskMonitor())
    return decompiled_function.getDecompiledFunction().getC()

def llm4dec(function):
    tokenizer, model = load_model()
    decompiled_function = decompile_function(function)
    inputs = tokenizer(decompiled_function, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=2048)
    return tokenizer.decode(outputs[0][len(inputs[0]):-1])

current_function = getFunctionContaining(currentAddress)
print(llm4dec(current_function))
