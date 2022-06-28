from torchgen.gen import (
    parse_native_yaml,
)

native_yaml_path = "./aten/src/ATen/native/native_functions.yaml"
tags_yaml_path = "./aten/src/ATen/native/tags.yaml"

parsed_yaml = parse_native_yaml(native_yaml_path, tags_yaml_path)
native_functions, backend_indices = (
    parsed_yaml.native_functions,
    parsed_yaml.backend_indices,
)

print("name,has_composite_kernel, has_composite_explicit_autograd_kernel,has_composite_implicit_autograd_kernel,is_abstract,is_view_op")
for n in native_functions:
    print(f'{n.root_name},{n.func.name},{n.has_composite_kernel},{n.has_composite_explicit_autograd_kernel},{n.has_composite_implicit_autograd_kernel},{n.is_abstract},{n.is_view_op}')


print(len(native_functions))
