import detectron2, inspect, os
print("detectron2 module file:", detectron2.__file__)
import detectron2.engine.train_loop as tl
print("train_loop path:", inspect.getsourcefile(tl))
print("PYTHONPATH:", os.environ.get("PYTHONPATH"))
