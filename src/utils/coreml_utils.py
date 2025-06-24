import time
import asyncio
import coremltools as ct
from termcolor import colored


compute_device_names = {
    ct.models.compute_device.MLNeuralEngineComputeDevice: colored("ane", "magenta"),
    ct.models.compute_device.MLCPUComputeDevice: colored("cpu", "blue"),
    ct.models.compute_device.MLGPUComputeDevice: colored("gpu", "green"),
}


async def print_compute_plan(
    mlmodelc_path: str,
    function_name="main",
    compute_unit=ct.ComputeUnit.CPU_AND_NE,
):
    start_time = time.time()
    compute_plan = await ct.models.ml_program.experimental.compute_plan_utils.load_compute_plan_from_path_on_device(
        mlmodelc_path,
        compute_units=compute_unit,
    )
    # compute_plan = asyncio.run(
    #     ct.models.ml_program.experimental.compute_plan_utils.load_compute_plan_from_path_on_device(
    #         mlmodelc_path,
    #         compute_units=compute_unit,
    #     )
    # )
    end_time = time.time()
    print("Compute plan loaded in", end_time - start_time, "seconds")
    program = compute_plan.model_structure.program
    function = program.functions[function_name]
    for operation in function.block.operations:
        # Get the compute device usage for the operation.
        compute_device_usage = (
            compute_plan.get_compute_device_usage_for_mlprogram_operation(operation)
        )
        if compute_device_usage is not None:
            preferred_device = compute_device_names[
                compute_device_usage.preferred_compute_device.__class__
            ]
            supported_devices = [
                compute_device_names[device.__class__]
                for device in compute_device_usage.supported_compute_devices
            ]
        else:
            preferred_device = ""
            supported_devices = [""]
        estimated_cost = compute_plan.get_estimated_cost_for_mlprogram_operation(
            operation
        )
        if estimated_cost is not None:
            estimated_cost = f"{estimated_cost.weight:.5f}"
        output_names = [output.name for output in operation.outputs]
        operation_name = operation.operator_name
        # formatted print in single line with tabulated constant spacing between values
        if operation_name != "const":
            # print(f"{operation_name}	Output: {output_names}	Preferred Device: {preferred_device}	Supported Devices: {supported_devices}	Estimated Cost: {estimated_cost}")
            print(
                f"{operation_name[:20]:<20} {",".join(output_names)[:60]:<60} {str(preferred_device)[:]:16} {str(",".join(supported_devices)):<30} {str(estimated_cost)[:15]:<10}"
            )


def print_compute_plan_sync(
    mlmodelc_path: str,
    function_name="main",
    compute_unit=ct.ComputeUnit.CPU_AND_NE,
):
    asyncio.run(
        print_compute_plan(
            mlmodelc_path,
            function_name,
            compute_unit,
        )
    )
