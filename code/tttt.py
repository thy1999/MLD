from safetensors import safe_open

tensors = {}
with safe_open("/public/home/dzhang/pyProject/hytian/XModel/Video-LLaVA-main/Video-LLaVA-7B/model-00001-of-00002.safetensors", framework="pt", device=0) as f:
    #import pdb;pdb.set_trace()
    metadata = f.metadata()
    # for k in f.keys():
    #     tensors[k] = f.get_tensor(k)
# print(tensors)
        


# # 仅加载部分张量（在多个GPU上运行时很有趣）
# from safetensors import safe_open

# tensors = {}
# with safe_open("model.safetensors", framework="pt", device=0) as f:
#     tensor_slice = f.get_slice("embedding")
#     vocab_size, hidden_dim = tensor_slice.get_shape()
#     tensor = tensor_slice[:, :hidden_dim]

