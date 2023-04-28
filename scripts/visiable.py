import netron


single = False
name = "single" if single else "multiply"  # 存放数据的文件夹名字

netron.start(f"../data/{name}/svr.onnx")