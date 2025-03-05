# 2.2　在conda环境编译优化TVM yolov3示例
# 示例：tvm_yolov3_optimize.py代码。
# （1）导入 numpy和 matplotlib
import numpy as np
import matplotlib.pyplot as plt
import sys
import ast

# （2）导入 tvm, relay
import tvm
from tvm import relay
from ctypes import *
#from tvm.contrib.download import download_testdata
from tvm.relay.testing.darknet import __darknetffi__
import tvm.relay.testing.yolo_detection
import tvm.relay.testing.darknet
import datetime
import os

# （3）设置 Model文件名
MODEL_NAME = 'yolov3'
##################################################

# （4）下载所需文件（本例实际上是将文件下载到本地，直接用本地下载的文件）
# -----------------------
# 程序下载cfg及权重文件
CFG_NAME = MODEL_NAME + '.cfg'
WEIGHTS_NAME = MODEL_NAME + '.weights'
#REPO_URL = 'https://github.com/dmlc/web-data/blob/master/darknet/'
#CFG_URL = REPO_URL + 'cfg/' + CFG_NAME + '?raw=true'
#WEIGHTS_URL = 'https://pjreddie.com/media/files/' + WEIGHTS_NAME
#cfg_path = download_testdata(CFG_URL, CFG_NAME, module="darknet")

# （5）直接用本地下载好的文件路径
cfg_path = "/home/jianming.wu/tvm/darknet-master/cfg/yolov3.cfg"
#weights_path = download_testdata(WEIGHTS_URL, WEIGHTS_NAME, module="darknet")
weights_path = "/home/jianming.wu/tvm/darknet-master/weights/yolov3.weights"
# 下载并加载DarkNet Library
if sys.platform in ['linux', 'linux2']:
    DARKNET_LIB = 'libdarknet2.0.so'
   #DARKNET_URL = REPO_URL + 'lib/' + DARKNET_LIB + '?raw=true'
elif sys.platform == 'darwin':
    DARKNET_LIB = 'libdarknet_mac2.0.so'
   #DARKNET_URL = REPO_URL + 'lib_osx/' + DARKNET_LIB + '?raw=true'
else:
    err = "Darknet lib is not supported on {} platform".format(sys.platform)
    raise NotImplementedError(err)
#lib_path = download_testdata(DARKNET_URL, DARKNET_LIB, module="darknet")
lib_path = "/home/jianming.wu/tvm/darknet-master/lib/libdarknet2.0.so"
# ******timepoint1-start*******
start1 = datetime.datetime.now()
# ******timepoint1-start*******
DARKNET_LIB = __darknetffi__.dlopen(lib_path)
net = DARKNET_LIB.load_network(cfg_path.encode('utf-8'), weights_path.encode('utf-8'), 0)
#net = DARKNET_LIB.load_network(cfg_path.encode('utf-8'), weights_path.encode('utf-8'), 0)
dtype = 'float32'
batch_size = 1
data = np.empty([batch_size, net.c, net.h, net.w], dtype)
shape_dict = {'data': data.shape}
print("Converting darknet to relay functions...")
#func, params = relay.frontend.from_darknet(net, dtype=dtype, shape=data.shape)
func, params = relay.frontend.from_darknet(net, dtype=dtype, shape=data.shape)

#s1 = ast.literal_eval(func)
#s2 = ast.literal_eval(params)
s1 = func
s2 = params
##################################################

# （6）将图导入Relay
# -------------------------
# 编译模型
#target = 'llvm'
#target_host = 'llvm'
#ctx = tvm.cpu(0)
target = tvm.target.Target("llvm", host="llvm")
dev = tvm.cpu(0)
#wujianming20210713start
# 优化
print("optimize relay graph...")
with tvm.relay.build_config(opt_level=2):
    func = tvm.relay.optimize(func, target, params)
# quantize
print("apply quantization...")

from tvm.relay import quantize
with quantize.qconfig():
   func = quantize.quantize(s1, s2)
#wujianming20210713finsih
print("Compiling the model...")
#print(func.astext(show_meta_data=False))
#with relay.build_config(opt_level=3):
# graph, lib, params = tvm.relay.build(func, target=target, params=params)
# Save the model
#tmp = util.tempdir()
#tmp =tempdir()
#lib_fname = tmp.relpath('model.tar')
#lib_fname =os.path.relpath('model.tar')
#lib.export_library(lib_fname)
#[neth, netw] = shape['data'][2:] # Current image shape is 608x608
#############################################
# Import the graph to Relay
# -------------------------
#target = tvm.target.Target("llvm", host="llvm")
#dev = tvm.cpu(0)
data = np.empty([batch_size, net.c, net.h, net.w], dtype)
shape = {"data": data.shape}
print("Compiling the model...")
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(func, target=target, params=params)
[neth, netw] = shape["data"][2:]  # Current image shape is 608x608
###############################################
# ******timepoint1-end*******
end1 = datetime.datetime.now()
# ******timepoint1-end*******
################################################

# （7）加载测试图片
# -----------------
test_image = 'dog.jpg'
print("Loading the test image...")
#img_url = REPO_URL + 'data/' + test_image + '?raw=true'
# img_path = download_testdata(img_url, test_image, "data")
img_path = "/home/jianming.wu/tvm/darknet-master/data/dog.jpg"
# ******timepoint2-start*******
start2 = datetime.datetime.now()
# ******timepoint2-start*******
data = tvm.relay.testing.darknet.load_image(img_path, netw, neth)
#############################################
# 在TVM上执行
# ----------------------
# 过程与其他示例没有差别
#from tvm.contrib import graph_runtime
from tvm.contrib import graph_executor

#m = graph_runtime.create(graph, lib, ctx)
m = graph_executor.GraphModule(lib["default"](dev))

# （8）设置输入
m.set_input('data', tvm.nd.array(data.astype(dtype)))
m.set_input(**params)
# 执行打印
print("Running the test image...")

m.run()

# （9）获得输出
tvm_out = []
if MODEL_NAME == 'yolov2':
    layer_out = {}
    layer_out['type'] = 'Region'
    # 获取区域层属性(n, out_c, out_h, out_w, classes, coords, background)
    layer_attr = m.get_output(2).asnumpy()
    layer_out['biases'] = m.get_output(1).asnumpy()
    out_shape = (layer_attr[0], layer_attr[1]//layer_attr[0],
                 layer_attr[2], layer_attr[3])
    layer_out['output'] = m.get_output(0).asnumpy().reshape(out_shape)
    layer_out['classes'] = layer_attr[4]
    layer_out['coords'] = layer_attr[5]
    layer_out['background'] = layer_attr[6]
    tvm_out.append(layer_out)

elif MODEL_NAME == 'yolov3':
    for i in range(3):
        layer_out = {}
        layer_out['type'] = 'Yolo'
        # 获取YOLO层属性 (n, out_c, out_h, out_w, classes, total)
        layer_attr = m.get_output(i*4+3).asnumpy()
        layer_out['biases'] = m.get_output(i*4+2).asnumpy()
        layer_out['mask'] = m.get_output(i*4+1).asnumpy()
        out_shape = (layer_attr[0], layer_attr[1]//layer_attr[0],
                     layer_attr[2], layer_attr[3])
        layer_out['output'] = m.get_output(i*4).asnumpy().reshape(out_shape)
        layer_out['classes'] = layer_attr[4]
        tvm_out.append(layer_out)

# （10） 检测, 并进行框选标记
thresh = 0.5
nms_thresh = 0.45
img = tvm.relay.testing.darknet.load_image_color(img_path)
_, im_h, im_w = img.shape
dets = tvm.relay.testing.yolo_detection.fill_network_boxes((netw, neth), (im_w, im_h), thresh, 1, tvm_out)
last_layer = net.layers[net.n - 1]
tvm.relay.testing.yolo_detection.do_nms_sort(dets, last_layer.classes, nms_thresh)
# ******timepoint2-end*******
end2 = datetime.datetime.now()
# ******timepoint2-end*******
#coco_name = 'coco.names'
#coco_url = REPO_URL + 'data/' + coco_name + '?raw=true'
#font_name = 'arial.ttf'
#font_url = REPO_URL + 'data/' + font_name + '?raw=true'
#coco_path = download_testdata(coco_url, coco_name, module='data')
#font_path = download_testdata(font_url, font_name, module='data')
coco_path = "/home/jianming.wu/tvm/darknet-master/data/coco.names"
font_path = "/home/jianming.wu/tvm/darknet-master/data/arial.ttf"
# ******timepoint3-start*******
start3 = datetime.datetime.now()
# ******timepoint3-start*******
with open(coco_path) as f:
    content = f.readlines()
names = [x.strip() for x in content]
tvm.relay.testing.yolo_detection.draw_detections(font_path, img, dets, thresh, names, last_layer.classes)
# ******timepoint3-end*******
end3 = datetime.datetime.now()
# ******timepoint3-end*******
print(end1-start1)
print(end2-start2)
print(end3-start3)
plt.imshow(img.transpose(1, 2, 0))
plt.show()

2.3　Python与C++调用关系
TVM包含很多的功能模块，这里介绍Python与C++的互相调用功能模块示例。不使用第三方的开源库（如boost.python、pybind11等），自主实现一套复杂，但精致、高效、强大的机制，具体包括以下3部分：
1）	最底层的C++数据结构支撑（围绕C++端的PackedFunc）。
2）	基于PackedFunc的函数注册（围绕TVM_REGISTER_GLOBAL）。
3）	偏上层的Python调用细节（围绕ctypes内置库和Python端PackedFunc）。
2.3.1　TVM中底层C++数据结构
1. 调用PackedFunc类
PackedFunc类是Python和C++互调的桥梁，此类实现代码在include/tvm/runtime/packed_func.h文件中，这里面还有一个TypedPackedFunc类，它只是PackedFunc类的一个wrapper（封装），主要增加了类型检查的功能，开发TVM的C++代码要尽可能使用这个类，但是为了把问题尽可能简化，只关注PackedFunc这个最底层类，其中用到了下面这几个关键的数据结构：
1）	TVMValue。
2）	TVMArgs。
3）	TVMPODValue_。
4）	TVMArgValue。
5）	TVMRetValue。
6）	TVMArgsSetter。
下面结合代码，逐个进行分析。
2. TVMValue存储数据类
这是最基本的一个数据结构，是一个union，实现在include/tvm/runtime/c_runtime_api.h，主要是为了存储C++和其他语言交互时所支持的几种类型的数据，代码很简单（其中DLDataType和DLDevice是两个复合数据类型，这里没有全部列出来，感兴趣的读者可去github查看相关细节）：
// include/tvm/runtime/c_runtime_api.h
typedef union {
  int64_t v_int64;
double v_float64;
  void* v_handle;
  const char* v_str;
  DLDataType v_type;
  DLDevice v_device;
} TVMValue;
3. TVMArgs参数实现类
这个类主要是为了封装传给PackedFunc的所有参数，这个类也比较简单原始，主要基于TVMValue、参数类型编码、参数个数来实现，代码如下所示：
class TVMArgs {
 public:
  const TVMValue* values;
  const int* type_codes;
  int num_args;
  TVMArgs(const TVMValue* values, 
          const int* type_codes, 
          int num_args) { ... }        
  inline int size() const { return num_args; }
  inline TVMArgValue operator[](int i) const { 
      return TVMArgValue(values[i], type_codes[i]); 
  }
};
4. TVMPODValue_结构类
这是一个内部使用的基类，主要服务于后面介绍到的TVMArgValue和TVMRetValue，从名字可以看出，这个类主要是处理POD类型的数据，POD是Plain Old Data的缩写。POD类型的实现核心是强制类型转换运算符重载（在C++中，类型的名字，包括类的名字本身也是一种运算符，即类型强制转换运算符），如下面代码所示：
class TVMPODValue_ {
 public:
  operator double() const { return value_.v_float64; }
  operator int64_t() const { return value_.v_int64; }
  operator void*() const { return value_.v_handle; }
  template <typename T>
  T* ptr() const { return static_cast<T*>(value_.v_handle); }
 protected:
  TVMValue value_;
  int type_code_;
};
5. TVMArgValue数据类型
这个类继承自前面的TVMPODValue_类，用作表示PackedFunc的一个参数，它和TVMPODValue_的区别是扩充了一些数据类型的支持，比如string、PackedFunc、TypedPackedFunc等，其中对后两个的支持是在C++代码中能够调用Python函数的根本原因。这个类只使用所保存的underlying data（源数据），而不会去做执行释放操作。代码如下所示：
class TVMArgValue : public TVMPODValue_ {
 public:
  TVMArgValue() {}
  TVMArgValue(TVMValue value, int type_code) 
  : TVMPODValue_(value, type_code) {}
  operator std::string() const {}
  operator PackedFunc() const { return *ptr<PackedFunc>(); }
  const TVMValue& value() const { return value_; }
  template <typename T>
  inline operator T() const;
  inline operator DLDataType() const;
  inline operator DataType() const;
};
6. TVMRetValue类型结构
这个类也是继承自TVMPODValue_类，主要作用是作为存放调用PackedFunc返回值的容器，它和TVMArgValue的区别是，它会管理所保存的underlying data，会对其做释放。这个类主要由以下四部分构成：
1）	构造函数与析构函数。
2）	对强制类型转换运算符重载的扩展。
3）	对赋值运算符的重载。
4）	辅助函数，包括释放资源的Clear函数。
具体代码如下所示：
class TVMRetValue: public TVMPODValue_ {
 public:
  // ctor与dtor, dtor将释放相关缓冲区
  TVMRetValue() {}
  ~TVMRetValue() { this->Clear(); }
  // conversion operators
  operator std::string() const { return *ptr<std::string>(); }
  operator DLDataType() const { return value_.v_type; }
  operator PackedFunc() const { return *ptr<PackedFunc>(); }
  // Assign operators
  TVMRetValue& operator=(double value) {}
  TVMRetValue& operator=(void* value) {}
  TVMRetValue& operator=(int64_t value) {}
  TVMRetValue& operator=(std::string value) {}
  TVMRetValue& operator=(PackedFunc f) {}  
 private:
  // judge type_code_, 发布基础数据
  void Clear() {
    if (type_code_ == kTVMStr || type_code_ == kTVMBytes) {
      delete ptr<std::string>();
    } else if(type_code_ == kTVMPackedFuncHandle) {
      delete ptr<PackedFunc>();
    } else if(type_code_ == kTVMNDArrayHandle) {
      NDArray::FFIDecRef(
        static_cast<TVMArrayHandle>(value_.v_handle));
    } else if(type_code_ == kTVMModuleHandle 
        || type_code_ == kTVMObjectHandle ) {
      static_cast<Object*>(value_.v_handle)->DecRef();
    }
    type_code_ = kTVMNullptr;
  }
};
7. TVMArgsSetter类型结构
这是一个用于给TVMValue对象赋值的辅助类，主要通过重载函数调用运算符来实现。实现代码如下所示：
class TVMArgsSetter {
 public:
  TVMArgsSetter(TVMValue* values, int* type_codes) 
    : values_(values), type_codes_(type_codes) {}
  void operator()(size_t i, double value) const {
    values_[i].v_float64 = value;
    type_codes_[i] = kDLFloat;
  }
  void operator()(size_t i, const string& value) const {
    values_[i].v_str = value.c_str();
    type_codes_[i] = kTVMStr;
  }
  void operator()(size_t i, const PackedFunc& value) const {
     values_[i].v_handle = const_cast<PackedFunc*>(&value);
     type_codes_[i] = kTVMPackedFuncHandle;
  }
 private:
  TVMValue* values_;
  int* type_codes_;
};
8. PackedFunc类型结构
有了前面所述的数据结构作为基础，再来看PackedFunc的实现。PackedFunc的实现很简单，内部只使用了一个存储函数指针的变量，再通过重载函数调用运算符来调用这个函数指针所指向的函数。实现代码如下所示：
class PackedFunc {
 public:
  using FType = function<void(TVMArgs args, TVMRetValue* rv)>;
  PackedFunc() {}
  explicit PackedFunc(FType body) : body_(body) {}

  template <typename... Args>
  inline TVMRetValue operator()(Args&&... args) const {
    const int kNumArgs = sizeof...(Args);
    const int kArraySize = kNumArgs > 0 ? kNumArgs : 1;
    TVMValue values[kArraySize];
    int type_codes[kArraySize];
    detail::for_each(TVMArgsSetter(values, type_codes), 
      std::forward<Args>(args)...);
    TVMRetValue rv;
    body_(TVMArgs(values, type_codes, kNumArgs), &rv);
    return rv;
  } 
  inline void CallPacked(TVMArgs args, TVMRetValue* rv) const {
    body_(args, rv);
  }
 private:
  FType body_;
};
　　总之，这里先调用PackedFunc，将参数传给TVMArgs，再用TVMRetValue返回。
2.3.2　进行函数注册

2. 几种注册接口
注册的函数可以是普通函数，也可以是lambda表达式，注册接口有三个：set_body、set_body_typed、set_body_method，第一个使用的是PackedFunc，后面两个使用的是TypedPackedFunc，PackedFunc在上节已经讲过了，TypedPackedFunc是PackedFunc的一个wrapper，实现比较复杂，以后有时间再细说这部分，下面举三个简单示例来展示下这三个注册接口的使用。
使用set_body接口注册lambda表达式（以src/topi/nn.cc中的topi.nn.relu为例）：
// src/topi/nn.cc
TVM_REGISTER_GLOBAL("topi.nn.relu")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = relu<float>(args[0]);
});
使用set_body_typed接口注册lambda表达式（以src/te/schedule/graph.cc中的schedule.PostDFSOrder为例）：
// src/te/schedule/graph.cc
TVM_REGISTER_GLOBAL("schedule.PostDFSOrder")
    .set_body_typed([](
          const Array<Operation>& roots, 
          const ReadGraph& g) {
      return PostDFSOrder(roots, g);
    });
使用set_body_method接口注册类内函数（以src/ir/module.cc中的ir.Module_GetGlobalVar为例）： 
// src/ir/module.cc
TVM_REGISTER_GLOBAL("ir.Module_GetGlobalVar")
    .set_body_method<IRModule>(&IRModuleNode::GetGlobalVar);
3. TVM_REGISTER_GLOBAL宏定义
这个宏定义在include/tvm/runtime/registry.h中，它的本质就是在注册文件定义了一个static的引用变量，引用到注册机内部new（新建）出来的一个新的Registry对象。实现代码如下所示：
// include/tvm/runtime/registry.h
#define TVM_REGISTER_GLOBAL(OpName)               
  static ::tvm::runtime::Registry& __mk_TVMxxx =  
      ::tvm::runtime::Registry::Register(OpName)  
上面的xxx其实是__COUNTER__这个编译器拓展宏生成的一个唯一标识符，GCC文档里对这个宏有详细的描述。
4. Registry::Manager注册方法
先来看最核心的Manager类，定义在src/runtime/registry.cc中，它是Registry的内部类，用来存储注册的对象。实现代码如下所示：
// src/runtime/registry.cc
struct Registry::Manager {
  static Manager* Global() {
    static Manager* inst = new Manager();
    return inst;
  }
  std::mutex mutex;
  unordered_map<std::string, Registry*> fmap;
};
这个数据结构很简单，从上面代码能得到下面几点信息：
1）	数据结构里面带锁，可以保证线程安全。
2）	Manager是个单例，限制类的实例化对象个数是一种技术，可以限制实例化对象个数为0个、1个、N个。
3）	使用unordered_map来存储注册信息，注册对象是Registry指针。
5. Registry注册类
这才是注册机的核心数据结构，定义在include/tvm/runtime/registry.h，简化过的代码如下（只保留了关键的数据结构和接口，原文使用了大量的模板、泛型等C++用法）。实现代码如下所示： 
// include/tvm/runtime/registry.hclass Registry {
 public:
  Registry& set_body(PackedFunc f);
  Registry& set_body_typed(FLambda f);
  Registry& set_body_method(R (T::*f)(Args...));
  static Registry& Register(const std::string& name);
  static const PackedFunc* Get(const std::string& name);
  static std::vector ListNames();
 protected:
  std::string name_;
  PackedFunc func_;
  friend struct Manager;
};
Registry的功能可以为三部分，相关的实现代码也比较简单，总结如下：
1）	设置注册函数的set_body系列接口，使用Registry的一系列set_body方法，可以把PackedFunc类型的函数对象设置到Registry对象中。
2）	创建Registry对象的Register静态接口，参照下面代码：
Registry& Registry::Register(const std::string& name) {
  Manager* m = Manager::Global();
  std::lock_guard<std::mutex> lock(m->mutex);
  Registry* r = new Registry();
  r->name_ = name;
  m->fmap[name] = r;
  return *r;
}
获取注册函数的Get静态接口，代码如下所示： 
const PackedFunc* Registry::Get(const std::string& name) {
  Manager* m = Manager::Global();
  std::lock_guard<std::mutex> lock(m->mutex);
  auto it = m->fmap.find(name);
  if (it == m->fmap.end()) return nullptr;
  return &(it->second->func_);
}
总之，在Python与C++间相互调用中，编译器会使用注册技术实现注册功能。本节内容相对简单，但是对于Python、C++的互调却至关重要，而且注册机也是一个被所有深度学习框架、编译器都会用到的技术，很有必要了解清楚。
2.3.3　上层Python调用
1. TVM用Python调用C++代码API原理
TVM使用Python的ctypes模块来调用C++代码提供的API，ctypes是Python内建的可以用于调用C/C++动态链接库函数的功能模块。
对于动态链接库提供的API，需要使用符合c语言编译和链接约定的API，因为Python的ctypes只和c兼容，而C++编译器会对函数和变量名进行name mangling，所以需要使用__cplusplus宏和extern "C"来得到符合c语言编译和链接约定的API，现以TVM给Python提供的接口为例。实现代码如下所示：
//TVM给Python提供的接口主要都在这个文件: 
// include/tvm/runtime/c_runtime_api.h
//下面主要展示了__cplusplus和extern "C"的用法，以及几个关键的API。
#ifdef __cplusplus
extern "C" {
#endif
 
int TVMFuncListGlobalNames(int* out_size, const char*** out_array...);
int TVMFuncGetGlobal(const char* name, TVMFunctionHandle* out...);
int TVMFuncCall(TVMFunctionHandle func, TVMValue* args, int* arg_type_codes, int num_args, TVMValue* ret_val, int* ret_type_code...);
    
#ifdef __cplusplus
}  // TVM_EXTERN_C
#endif
2. 加载TVM动态库
TVM的Python代码从python/tvm/__init__.py中开始真正执行。实现代码如下所示：
from ._ffi.base import TVMError, __version__
# 这句简单的import代码，会执行python/tvm/_ffi/__init__.py：
from .base import register_error
from .registry import register_func
from .registry import _init_api, get_global_func
# 上面的第一句，会导致python/tvm/_ffi/base.py中的下面代码被执行：
def _load_lib():
    lib = ctypes.CDLL(lib_path[0], ctypes.RTLD_GLOBAL)
    return lib, os.path.basename(lib_path[0])
_LIB, _LIB_NAME = _load_lib()
上面的lib_path[0]是TVM动态链接库的全路径名称，在linux系统做的试验，链接库的名称是/xxx/libtvm.so（不同的系统动态库的名字会有所不同，windows系统是.dll，苹果系统是.dylib，linux系统是.so），在_load_lib函数执行完成后，_LIB和_LIB_NAME都完成了初始化，其中_LIB是一个ctypes.CDLL类型的变量，可以认为它是能够操作TVM动态链接库的export symbols的一个全局句柄，_LIB_NAME是libtvm.so这个字符串。这样后续在Python中，就能通过_LIB这个桥梁不断地和C++的部分进行交互。 
3. Python关联C++的PackedFunc
前面已经对C++中的PackedFunc做了详细的剖析，这里主要来厘清楚Python的代码中是怎么使用这个核心组件的，还是通过代码，一步步来看。
Python中来获取C++API的底层函数是_get_global_func。实现代码如下所示：
# python/tvm/_ffi/_ctypes/packed_func.py
def _get_global_func(func_name):
    handle = ctypes.c_void_p()
    _LIB.TVMFuncGetGlobal(c_str(name), ctypes.byref(handle))
    return _make_packed_func(handle, False)
这里面handle是一个相当于void类型的指针变量，因为从ctypes的官方文档中可以查到，c_void_p对应的primitive C compatible data type。实现代码如下所示：
// src/runtime/registry.cc
int TVMFuncGetGlobal(const char* name, TVMFunctionHandle* out) {
  const tvm::runtime::PackedFunc* fp 
      = tvm::runtime::Registry::Get(name);
  *out = new tvm::runtime::PackedFunc(*fp);
}
与C++PackedFunc的关联工作这时候才完成一半，在_get_global_func的最后调用了_make_packed_func这个函数。实现代码如下所示：
# python/tvm/_ffi/_ctypes/packed_func.py
def _make_packed_func(handle, is_global):
    obj = PackedFunc.__new__(PackedFuncBase)
    obj.is_global = is_global
    obj.handle = handle
    return obj
可以看到_make_packed_func函数中创建了一个定义在python/tvm/runtime/packed_func.py中的Python PackedFunc对象，PackedFunc其实是一个空实现，它继承自PackedFuncBase类，PackedFuncBase类中定义了一个__call__函数。实现代码如下所示：
# python/tvm/_ffi/_ctypes/packed_func.py
class PackedFuncBase(object):
  def __call__(self, *args):
    values, tcodes, num_args = _make_tvm_args(args, temp_args)
    ret_val = TVMValue()
    ret_tcode = ctypes.c_int()
    _LIB.TVMFuncCall(
        self.handle,
        values,
        tcodes,
        ctypes.c_int(num_args),
        ctypes.byref(ret_val),
        ctypes.byref(ret_tcode),
    )
    return ret_val
从上面可以看出，Python的__call__函数调用了C的TVMFuncCall这个API，把前面保存有C++ PackedFunc对象地址的handle以及相关的函数参数传了进去，TVMFuncCall的主体代码如下所示：
// src/runtime/c_runtime_api.cc
int TVMFuncCall(TVMFunctionHandle handle, TVMValue* args, ...)
  (*static_cast<const PackedFunc*>(handle))
      .CallPacked(TVMArgs(args, arg_type_codes, num_args), &rv);
}
这样就完成了把C++中的PackedFunc映射到了Python中的PackedFunc，在Python代码中只需要调用Python中创建好的PackedFunc对象，就会通过上面分析的过程来一步步调到C++的代码中。
4. 把注册关联到Python
1）	实现代码如下所示：
# python/tvm/runtime/_ffi_api.py
tvm._ffi._init_api("runtime", __name__)
# python/tvm/relay/op/op.py
tvm._ffi._init_api("relay.op", __name__)
# python/tvm/relay/backend/_backend.py
tvm._ffi._init_api("relay.backend", __name__)
5. 从Python映射到C++示例分析
以TVM中求绝对值的函数abs为例，这个函数实现在tir模块，函数的功能很简单，不会造成额外的理解负担，只关注从Python调用是怎么映射到C++中的，先看在C++中abs函数的定义和注册。执行代码如下所示： 
// src/tir/op/op.cc
# 函数定义
PrimExpr abs(PrimExpr x, Span span) { ... }
# 函数注册
TVM_REGISTER_GLOBAL("tir.abs").set_body_typed(tvm::abs);
# 再看Python端调用 
# python/tvm/tir/_ffi_api.py
# C++ tir注册函数Python PackedFunc
# 关联_ffi_api模块
tvm._ffi._init_api("tir", __name__)
# python/tvm/tir/op.py
# 定义绝对值abs的Python函数
# 关联_ffi_api的Python PackedFunc对象
def abs(x, span=None):
    return _ffi_api.abs(x, span)
# 使用函数实现代码如下所示： 
import tvm
from tvm import tir

rlt = tir.abs(-100)
print("abs(-100) = %d" % (rlt)
2.4　 TVM自定义代码示例
1. 如何补全Python接口
测试代码中可使用字符串解析的方式，但是从其他tutorial中发现，还存在一种tvm.target.cuda()的设备建立方式，这个很明显比字符串解析相对找起来容易（字符串最终对应的也是这种方式）。按照这种方式找到了tvm/python/tvm/target.py文件中，这个类中定义了现在能支持的target，而添加新的target叫作dpu。实现代码如下所示：
def dpu(model='unknown', options=None):
    // Returns a dpu 目标.
    Parameters
    ----------
    model: str
        The model of dpu device
    options : str or list of str
        Additional options
    
    opts = _merge_opts(['-model=%s' % model], options)
return _api_internal._TargetCreate("dpu", *opts)
每个设备都包括硬件自身的上下文信息和硬件上运行软件运行时也就是runtime，在TVM中相关的软件运行时信息在tvm/python/tvm/_ffi/runtime_ctypes.py文件中，添加对dpu的支持在class TVMContext的两个掩码MASK2STR和STR2MASK中，分别用以下所示添加：
13: 'dpu'，
与
'dpu':13。
2. Python与C交互接口
实现代码如下所示：
TVM_REGISTER_API("_TargetCreate")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  std::string target_name = args[0];
  std::vector<std::string> options;
  for (int i = 1; i < args.num_args; ++i) {
    std::string arg = args[i];
    options.push_back(arg);
  }
  *ret = CreateTarget(target_name, options);
  });
这段代码的意思就是通过一种TVM_REGISTER_API的注册机制，注册_TargetCreate函数，真正的函数体是.set_body内执行的，实际上C++中tvm::CreateTarget函数。TVM_REGISTER_API的注册机制在TVM项目中非常普遍，其实现在项目中也有，不是主要的研究内容，不需要改，所以不另行叙述。 
3. 正确维护中间代码的IR pass变换中新设备引入的特性
上边提到，在src/codegenCodeGen/build_module.cc文件中的tvm::CreateTarget函数中添加对dpu的支持。实现代码如下所示：
else if (target_name == "dpu") {
    t->device_type = kDLDPU;
  }
这里边的kDLDPU是一个DLDeviceType类型值，实现是在3rdparty/dlpack/include/dlpack/dlpack.h中添加的。
	kDLDPU =13,
同时在include/tvm/runtime/device_api.h：200补充对kDLDPU的支持：
case kDLDPU: return "dpu";
Target部分添加完了，还需要补充运行时的内容。运行时的内容在src/runtime/目录下，需要在module.cc中添加对dpu 的支持。在RuntimeEnabled函数中，增加以下代码：
else if (target == "dpu") {
    f_name = "device_api.dpu";
  }
这只是添加了一个名字的支持，还需要新建一个dpu目录，里边存放DPUModuleNode、DPUWorkspace等支持，测试代码的getSource函数的真正实现也存放在这里边，主要模仿CUDA和openCl的实现进行。目前存放有dpu_common.h、dpu_device_api.cc、dpu_module.cc、dpu_module.h四个文件，大概1K行代码，实现逻辑也不是很复杂。
4. 如何支持代码生成
实现代码如下所示：
Target DPU(const std::vector<std::string>& options ) {
  return CreateTarget("dpu", options);
}
实现代码如下所示：
runtime::Module BuildDPU(Array<LoweredFunc> funcs) {
  using tvm::runtime::Registry;
  bool output_ssa = false;
  CodeGenCodeGenDPU cg;
  cg.Init(output_ssa);
  for (LoweredFunc f: funcs) {
    cg.AddFunction(f);
  }
  std::string code = cg.Finish();
  if (const auto* f = Registry::Get("tvm_callback_dpu_postproc")) {
    code = (*f)(code).operator std::string();
  }
  return DPUModuleCreate(code, "dpu", ExtractFuncInfo(funcs), code);
}
TVM_REGISTER_API("codegenCodeGen.build_dpu")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = BuildDPU(args[0]);
  });
5. 如何添加编译选项支持
上边可以说是完成了从设备添加到代码生成的部分，但是如果只有上边的话，新添加的设备一直无法运行。但如果仅是对一个设备进行修改的话，这部分并没有必要。后来排查发现是部分代码未编译进去导致的。所以开始修改cmake配置。
在上一个TVM调试文档中提到，编译需要打开LLVM和CUDA选项，这里新添加了dpu的设备，需要增加一个新的编译选项，在cmake/config.cmake中添加。实现代码如下所示：
#Build DPU
set(USE_DPU ON)
这部分的配置代码相对比较简单，就是指定runtime对应的目录。
# DPU Module
if(USE_DPU)
  message(STATUS "Build with DPU support")
  file(GLOB RUNTIME_DPU_SRCS src/runtime/dpu/*.cc)
  list(APPEND RUNTIME_SRCS ${RUNTIME_DPU_SRCS})
else()
  message(STATUS "NOT BUILD DPU SUPPORT")
endif(USE_DPU)
这里修改完config.cmake，需要重新拷贝到build目录下，以使下次配置生效。在上边也提到了，编译tvm时是cmake目录下的config.cmake和CMakeLists.txt共同工作生效。在CMakeLists.txt中添加。实现代码如下所示：
tvm_option(USE_DPU "Build with DPU" ON)
include(cmake/modules/DPU.cmake)
然后在build目录下，运行cmake命令，重新编译生效。
cmake  -DCMAKE_BUILD_TYPE=Debug ../
make
这里不加-DCMAKE_BUILD_TYPE=Debug的话，C++代码无法进行调试。
2.4.2　TVM代码生成实现示例
1. 如何使用relay.build
实现代码如下所示：
# Build with Relay
with relay.build_config(opt_level=0):
graph, lib, params = relay.build(func, target, params=params)
2. 如何调试跟踪代码
怎样调试代码，vscode在函数上用Alt键，再单击跳转按钮Alt+单击跳转，用pycallgraph进行可视化（用pip安装），再用GCN编译。实现代码如下所示：
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
from pycallgraph import Config
graphviz = GraphvizOutput()
graphviz.output_file = 'relay_callgraph.png'
config = Config(max_depth=5)
with PyCallGraph(output=graphviz,config=config):
# 用Relay编译
    with relay.build_config(opt_level=0):
graph, lib, params = relay.build(func, target, params=params)
    归纳起来，包括以下步骤：
1）	实现函数调用，例如，tvm.relay.build_module.build->tvm.relay.build_module.BuildModule.build
2）	实现FFI打包调用，如C++与Python互调。
3）	标注的结点（执行时间长）是关键路径。
4）	如tvm.build_module.lower调用结点14次，类似Relay算子，用Relay IR进行计算图可视化。
跟踪relay.build模块，转到python/tvm/relay/build_module.py（在relay/__init__.py中，将build直接import到relay，绕过build_module层），这里build是build_module全局函数（helper）。实现代码如下所示：
def build(mod, target=None, target_host=None, params=None):
# do somthing
if isinstance(autotvm.DispatchContext.current, autotvm.FallbackContext):
tophub_context = autotvm.tophub.context(list(target.values()))
else:
tophub_context = autotvm.util.EmptyContext()
with tophub_context:
bld_mod = BuildModule()
graph_json, mod, params = bld_mod.build(func, target, target_host, params)
return graph_json, mod, params
寻找AutoTVM是否有tune记录，先构造tophub_context，再构建BuildModule，然后跳转到BuildModule.build中，最后返回到BuildModule.__init__中。实现代码如下所示：
class BuildModule(object):
// Build a Relay function to run on TVM graph runtime. This class is used
// to expose the `RelayBuildModule` APIs implemented in C++.构建一个Relay函数
// 以便在TVM图形运行时上运行

def __init__(self):
self.mod = _build_module._BuildModule()
self._get_graph_json = self.mod["get_graph_json"]
self._get_module = self.mod["get_module"]
self._build = self.mod["build"]
self._optimize = self.mod["optimize"]
self._set_params_func = self.mod["set_params"]
self._get_params_func = self.mod["get_params"]
def build(self, func, target=None, target_host=None, params=None):
target = _update_target(target)
# Setup the params.
        if params:
self._set_params(params)
# Build the function
          self._build(func, target, target_host)
# Get artifacts
          graph_json = self.get_json()
mod = self.get_module()
params = self.get_params()
return graph_json, mod, params
_build_module._BuildModule()通过FFI模块，在python/tvm/relay/_build_module.py中，与C++建立联系（tvm._ffi._cytpes.function.Function.__call__）。实现代码如下所示：
from tvm._ffi.function import _init_api
_init_api("relay.build_module", __name__)
// 对应C++在src/relay/backend/build_module.cc中
runtime::Module RelayBuildCreate() {
auto exec = make_object<RelayBuildModule>();
return runtime::Module(exec);
}
TVM_REGISTER_GLOBAL("relay.build_module._BuildModule")
.set_body([](TVMArgs args, TVMRetValue* rv) {
*rv = RelayBuildCreate();
});
先注册RelayBuildModule，再在RelayBuildModule中搜索build函数。实现代码如下所示：
PackedFunc GetFunction(const std::string& name,
const ObjectPtr<Object>& sptr_to_self) final {
if (name == "build") {
return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
CHECK_EQ(args.num_args, 3);
this->Build(args[0], args[1], args[2]);
});
}
使用this->Build，指向BuildRelay。实现代码如下所示：
void BuildRelay(
Function func,
const std::unordered_map<std::string, tvm::runtime::NDArray>& params) {
// Optimize input Relay Function and returns Relay Module
relay::Module relay_module = Optimize(func, targets_, params);
// Get the updated function.
func = relay_module->Lookup("main");
// Generate code for the updated function.
graph_codegenCodeGen_ = std::unique_ptr<GraphCodegenCodeGen>(new GraphCodegenCodeGen());
graph_codegenCodeGen_->Init(nullptr, targets_);
graph_codegenCodeGen_->CodegenCodeGen(func);
ret_.graph_json = graph_codegenCodeGen_->GetJSON();
ret_.params = graph_codegenCodeGen_->GetParams();
auto lowered_funcs = graph_codegenCodeGen_->GetLoweredFunc();
if (lowered_funcs.size() == 0) {
LOG(WARNING) << "no lowered funcs exist in the compiled module";
} else {
ret_.mod = tvm::build(
lowered_funcs,
target_host_,
BuildConfig::Current());
}
}
转到build核中，进行如下具体模块工作：
1）	优化。
2）	计算图生成。
3）	后端代码生成。
3. 如何进行图优化张量计算
实现代码如下所示：
relay::Module Optimize(
Function func,
const TargetsMap& targets,
const std::unordered_map<std::string, runtime::NDArray>& params) {
// BindParamsByName(func, params)
// Perform Module->Module optimizations.
relay::Module relay_module = relay::ModuleNode::FromExpr(func);
Array<Pass> pass_seqs;
// 运行所有语言合法化passes.
// ...
pass_seqs.push_back(transform::SimplifyInference());
//
// ...fskip
//
pass_seqs.push_back(transform::EliminateCommonSubexpr(fskip));
pass_seqs.push_back(transform::CombineParallelConv2DConv2d(3));
pass_seqs.push_back(transform::CombineParallelDense(3));
pass_seqs.push_back(transform::FoldConstant());
pass_seqs.push_back(transform::FoldScaleAxis());
pass_seqs.push_back(transform::CanonicalizeCast());
pass_seqs.push_back(transform::CanonicalizeOps());
// ...AlterOpLayout
pass_seqs.push_back(transform::FoldConstant());
// Create a sequential pass and perform optimizations.
transform::Pass seq = transform::Sequential(pass_seqs);
// ... judge & do
relay_module = seq(relay_module);
// 处理异构编译
transform::PassContext pass_ctx = PassContext::Current();
if (targets_.size() > 1) {
relay_module =
RunDeviceAnnotationPass(relay_module, pass_ctx->fallback_device);
}
// 如果需要，可进行融合操作
relay_module = transform::FuseOps()(relay_module);
relay_module = transform::InferType()(relay_module);
CHECK(relay_module.defined());
return relay_module;
}
5. 如何生成后端代码
实现代码如下所示：
// 构建异构执行
runtime::Module build(const Map<Target, Array<LoweredFunc>>& inputs,
const Target& target_host,
const BuildConfig& config) {
Array<LoweredFunc> fhost_all;
std::vector<runtime::Module> device_modules;
Target target_host_val = target_host;
if (!target_host.defined()) {
for (const auto& it : inputs) {
if (it.first->device_type == kDLCPU) {
target_host_val = it.first;
break;
}
}
}
if (!target_host_val.defined()) {
target_host_val = DefaultTargetHost(target_host_val);
}
for (const auto& it : inputs) {
auto host_dev_funcs =
split_dev_host_funcs(it.second, it.first, target_host_val, config);
auto& fhost = host_dev_funcs[0];
auto& fdevice = host_dev_funcs[1];
// 获取特定目标的模块
runtime::Module mdev = DeviceBuild(fdevice, it.first);
for (const auto& it : fhost) {
fhost_all.push_back(it);
}
device_modules.push_back(mdev);
}
runtime::Module mhost = codegenCodeGen::Build(fhost_all, target_host_val->str());
// 导入所有模块
for (const auto& it : device_modules) {
if (it.operator->()) {
mhost.Import(it);
}
}
return mhost;
}
// 在mhost = codegenCodeGen::Build核中，进行代码生成（src/codegenCodeGen/codegenCodeGen.cc）。
runtime::Module Build(const Array<LoweredFunc>& funcs,
const std::string& target) {
// do something
std::string build_f_name = "codegenCodeGen.build_" + mode;
// the build function.
const PackedFunc* bf = runtime::Registry::Get(build_f_name);
runtime::Module m = transformed_funcs.empty() ?
(*bf)(funcs, target) :
(*bf)(transformed_funcs, target);
return m;
}
6. 如何进行tvm.build配置
用tvm.build进行编译，按以下所示代码实现：
s = tvm.create_schedule(C.op)
tgt = "llvm" # "cuda"
fadd = tvm.build(s,[A,B,C],target=tgt,name="myadd")
用tvm.build后，先转至python/tvm/build_module.py，再使用build执行2步。这里第一步生成降级下译高层代码，第二步完成设备代码生成。
7. 如何进行代码转换
实现代码如下所示： 
flist = lower(inputs, args, name=name, binds=binds)
lower在python/tvm/build_module.py中，对应relay.build中Optimize，进行operator-level优化。
def lower(sch,
args,
name="default_function",
binds=None,
simple_mode=False):
# initialization
# Phase 0
if isinstance(sch, schedule.Schedule):
stmt = form_body(sch)
for f in lower_phase0:
stmt = f(stmt)
compact = ir_pass.VerifyCompactBuffer(stmt)
binds, arg_list = get_binds(args, compact, binds)
# Phase 1
stmt = ir_pass.RewriteForTensorCore(stmt, sch, binds)
stmt = ir_pass.StorageFlatten(stmt, binds, 64, cfg.instrument_bound_checkers)
stmt = ir_pass.CanonicalSimplify(stmt)
for f in lower_phase1:
stmt = f(stmt)
# Phase 2
if not simple_mode:
stmt = ir_pass.LoopPartition(stmt, cfg.partition_const_loop)
if cfg.disable_vectorize:
stmt = ir_pass.SkipVectorize(stmt)
else:
stmt = ir_pass.VectorizeLoop(stmt)
stmt = ir_pass.InjectVirtualThread(stmt)
stmt = ir_pass.InjectDoubleBuffer(stmt, cfg.double_buffer_split_loop)
stmt = ir_pass.StorageRewrite(stmt)
stmt = ir_pass.UnrollLoop(
stmt,
cfg.auto_unroll_max_step,
cfg.auto_unroll_max_depth,
cfg.auto_unroll_max_extent,
cfg.unroll_explicit)
for f in lower_phase2:
stmt = f(stmt)
# Phase 3
stmt = ir_pass.Simplify(stmt)
stmt = ir_pass.RemoveNoOp(stmt)
if not cfg.disable_select_rewriting:
stmt = ir_pass.RewriteUnsafeSelect(stmt)
for f in lower_phase3:
stmt = f(stmt)
# Instrument BoundCheckers
if cfg.instrument_bound_checkers:
stmt = ir_pass.InstrumentBoundCheckers(stmt)
if simple_mode:
return stmt
return ir_pass.MakeAPI(stmt, name, arg_list, 0, cfg.restricted_func)# Phase 0
if isinstance(sch, schedule.Schedule):
stmt = form_body(sch)
for f in lower_phase0:
stmt = f(stmt)
compact = ir_pass.VerifyCompactBuffer(stmt)
binds, arg_list = get_binds(args, compact, binds)
# Phase 1
stmt = ir_pass.RewriteForTensorCore(stmt, sch, binds)
stmt = ir_pass.StorageFlatten(stmt, binds, 64, cfg.instrument_bound_checkers)
stmt = ir_pass.CanonicalSimplify(stmt)
for f in lower_phase1:
stmt = f(stmt)
# Phase 2
if not simple_mode:
stmt = ir_pass.LoopPartition(stmt, cfg.partition_const_loop)
if cfg.disable_vectorize:
stmt = ir_pass.SkipVectorize(stmt)
else:
stmt = ir_pass.VectorizeLoop(stmt)
stmt = ir_pass.InjectVirtualThread(stmt)
stmt = ir_pass.InjectDoubleBuffer(stmt, cfg.double_buffer_split_loop)
stmt = ir_pass.StorageRewrite(stmt)
stmt = ir_pass.UnrollLoop(
stmt,
cfg.auto_unroll_max_step,
cfg.auto_unroll_max_depth,
cfg.auto_unroll_max_extent,
cfg.unroll_explicit)
for f in lower_phase2:
stmt = f(stmt)
# Phase 3
stmt = ir_pass.Simplify(stmt)
stmt = ir_pass.RemoveNoOp(stmt)
if not cfg.disable_select_rewriting:
stmt = ir_pass.RewriteUnsafeSelect(stmt)
for f in lower_phase3:
stmt = f(stmt)
# Instrument BoundCheckers
if cfg.instrument_bound_checkers:
stmt = ir_pass.InstrumentBoundCheckers(stmt)
if simple_mode:
return stmt
return ir_pass.MakeAPI(stmt, name, arg_list, 0, cfg.restricted_func)
在src/api/api_pass.cc中，用tvm.ir_pass进行注册（C++在tvm里C++中，搜索ir_pass的API）。

2.5.1　配置张量与创建调度
1. 从张量开始 
执行代码如下所示：
import tvm
import numpy as np

# 张量表示
# args: (shape, label)
A = tvm.placeholder((10,), name='A')
B = tvm.placeholder((10,), name='B')
# args: (shape, function, label)
# function represented in lambda expression (element-wise)
# lambda axis1, axis2, ... : f(axis1, axis2, ...)
C = tvm.compute((10,), lambda i: A[i] + B[i], name="C")

# 创建调度
s = tvm.create_schedule(C.op)
# print low level codes
print(tvm.lower(s,[A,B,C],simple_mode=True))
// tvm定义compute，各维度A，B对应值相加。schedule为s生成嵌套循环，打印
// 输出降级（Lowering）下译（Lowering）代码验证。
for (i: int32, 0, 10) {
  C_2[i] = ((float32*)A_2[i] + (float32*)B_2[i])
}
// 优化schedule，如划分内外循环，用split实现。
# split(parent[, factor, nparts])
// 根据提供外部范围的因子分解，返回迭代的外部值和内部值。
bx, tx = s[C].split(C.op.axis[0],factor=2)
print(tvm.lower(s,[A,B,C],simple_mode=True))
// 打印降级（Lowering）下译（Lowering）代码，将单层循环划分为内外循环。
for (i.outer: int32, 0, 5) {
    for (i.inner: int32, 0, 2) {
      C_2[((i.outer*2) + i.inner)] = ((float32*)A_2[((i.outer*2) + i.inner)] + (float32*)B_2[((i.outer*2) + i.inner)])
    }
  }
// 调度张量表示。配置target与target_host，调用build设备代码。
tgt_host = "llvm"
// 如果GPU已启用，可更改为相应的GPU，例如：cuda、opencl、rocm
tgt = "llvm" # cuda llvm
n = 10
fadd = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name="myadd")
ctx = tvm.context(tgt,0)
a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), ctx)
c = tvm.nd.array(np.zeros(n,dtype=C.dtype), ctx)
fadd(a,b,c) # run
# test
tvm.testing.assert_allclose(c.asnumpy(),a.asnumpy() + b.asnumpy())
print(fadd.get_source())
2. 如何用tvm.build生成设备代码
实现代码如下所示：
def build(inputs,args=None,target=None,target_host=None,name="default_function",binds=None):
    if isinstance(inputs, schedule.Schedule):
        if args is None:
            raise ValueError("args must be given for build from schedule")
        input_mod = lower(inputs, args,name=name,binds=binds)
    // skip some code.....
// 整理成如下形式: 
target_input_mod = {'target': [LoweredFunc]}
2.5.2　如何进行降级下译算子优化
实现代码如下所示：
def lower(sch,args,name="main",binds=None,simple_mode=False):
    # config setup
    pass_ctx = PassContext.current()
    instrument_bound_checkers = bool(pass_ctx.config.get(
        "tir.instrument_bound_checkers", False))
    disable_vectorize = bool(pass_ctx.config.get(
        "tir.disable_vectorize", False))
    add_lower_pass = pass_ctx.config.get("tir.add_lower_pass", [])
    lower_phase0 = [x[1] for x in add_lower_pass if x[0] == 0]
    lower_phase1 = [x[1] for x in add_lower_pass if x[0] == 1]
    lower_phase2 = [x[1] for x in add_lower_pass if x[0] == 2]
    lower_phase3 = [x[1] for x in add_lower_pass if x[0] > 2]
    # Phase 0
    if isinstance(sch, schedule.Schedule):
        mod = form_irmodule(sch, args, name, binds)
    else:
        mod = sch
    pass_list = lower_phase0
    # Phase 1
    pass_list += [
        tvm.tir.transform.InjectPrefetch(),
        tvm.tir.transform.StorageFlatten(64, instrument_bound_checkers),
        tvm.tir.transform.BF16Legalize(),
        tvm.tir.transform.NarrowDataType(32),
        tvm.tir.transform.Simplify(),
    ]
    pass_list += lower_phase1
    # Phase 2
    if not simple_mode:
        pass_list += [(tvm.tir.transform.LoopPartition())]
    pass_list += [
        tvm.tir.transform.VectorizeLoop(not disable_vectorize),
        tvm.tir.transform.InjectVirtualThread(),
        tvm.tir.transform.InjectDoubleBuffer(),
        tvm.tir.transform.StorageRewrite(),
        tvm.tir.transform.UnrollLoop()
    ]
    pass_list += lower_phase2
    # Phase 3
    pass_list += [
        tvm.tir.transform.Simplify(),
        tvm.tir.transform.RemoveNoOp(),
    ]
    pass_list += [tvm.tir.transform.RewriteUnsafeSelect()]
    pass_list += [tvm.tir.transform.HoistIfThenElse()]
    pass_list += lower_phase3
    # Instrument BoundCheckers
    if instrument_bound_checkers:
        pass_list += [tvm.tir.transform.InstrumentBoundCheckers()]
    optimize = tvm.transform.Sequential(pass_list)
    mod = optimize(mod)
    return mod

2.5.3　如何构建host目标程序
实现代码如下所示：
def build(inputs, args=None, target=None, target_host=None, name="default_function", binds=None):
    # 跳过一些代码.....
    device_modules = []
    for tar, input_mod in target_input_mod.items():
       # build for device module
        mod_host, mdev = _build_for_device(input_mod, tar, target_host)
        mod_host_all.update(mod_host)
        device_modules.append(mdev)
    # 生成统一的主机模块
    rt_mod_host = codegenCodeGen.build_module(mod_host_all, target_host)
    # Import all modules.
    for mdev in device_modules:
        if mdev:
            rt_mod_host.import_module(mdev)
    return rt_mod_host
2.5.4　如何实现后端代码生成
实现代码如下所示：
runtime::Module Build(IRModule mod, Target target) {
  if (transform::PassContext::Current()
          ->GetConfig<Bool>("tir.disable_assert", Bool(false))
          .value()) {
    mod = tir::transform::SkipAssert()(mod);
  }
  std::string build_f_name;
  if (target->kind->name == "micro_dev") {
    build_f_name = "target.build.c";
  } else {
    build_f_name = "target.build." + target->kind->name;
  }
  // the build function.
  const PackedFunc* bf = runtime::Registry::Get(build_f_name);
  CHECK(bf != nullptr) << build_f_name << "is not enabled";
  return (*bf)(mod, target);
}
TVM_REGISTER_GLOBAL("target.Build").set_body_typed(Build);
 
第3章　算子融合与图优化
3.3　图融合GCN示例    
基于稀疏矩阵乘法，GCN图卷积层实现代码如下所示：
import torch
    import torch.nn as nn

    class GraphConvolution(nn.Module):
        // GCN layer
        def __init__(self, in_features, out_features, bias=True):
            super(GraphConvolution, self).__init__()
            self.in_features=in_features
            self.out_features=out_features
            self.weight=nn.Parameter(torch.Tensor(in_features, out_features))
            if bias:
                self.bias=nn.Parameter(torch.Tensor(out_features))
            else:
                self.register_parameter('bias', None)
            self.reset_parameters()
        def reset_parameters(self):
            nn.init.kaiming_uniform_(self.weight)
            if self.bias isnotNone:
                nn.init.zeros_(self.bias)
        def forward(self, input, adj):
            support=torch.mm(input, self.weight)
            output=torch.spmm(adj, support)
            if self.bias isnotNone:
                return output+self.bias
            else:
                return output
        def extra_repr(self):
            return'in_features={}, out_features={}, bias={}'.format(
                self.in_features, self.out_features, self.bias isnotNone
          )
对于GCN，只需要将图卷积层堆积起来就可以，这里实现一个两层的GCN。实现代码如下所示：
class GCN(nn.Module):
    // 两层GCN的简单示例
    def __init__(self, nfeat, nhid, nclass):
        super(GCN, self).__init__()
        self.gc1=GraphConvolution(nfeat, nhid)
        self.gc2=GraphConvolution(nhid, nclass)
    def forward(self, input, adj):
        h1=F.relu(self.gc1(input, adj))
        logits=self.gc2(h1, adj)
        return logits
这里的激活函数采用ReLU，后面，将用这个网络实现一个图中节点的半监督分类任务。
3.3.2　融合BN与Conv层
实现代码如下所示：
import torch
    import torchvision
    
    def fuse(conv, bn):
        fused=torch.nn.Conv2dConv2d(
            conv.in_channels, 
            conv.out_channels, 
            kernel_size=conv.kernel_size, 
            stride=conv.stride, 
            padding=conv.padding, 
            bias=True
      )
   
        # 设置权重
        w_conv=conv.weight.clone().view(conv.out_channels, -1)
        w_bn=torch.diag(bn.weight.div(torch.sqrt(bn.eps+bn.running_var)))
        fused.weight.copy_( torch.mm(w_bn, w_conv).view(fused.weight.size()))
        
        # 设置bias
        if conv.bias isnotNone:
            b_conv=conv.bias
        else:
            b_conv=torch.zeros( conv.weight.size(0))
        b_bn=bn.bias - bn.weight.mul(bn.running_mean).div(
                              torch.sqrt(bn.running_var+bn.eps)
                          )
        fused.bias.copy_( b_conv+b_bn)
        return fused
    
    # 测试
    # 需要关闭梯度计算，因为没有编写
    torch.set_grad_enabled(False)
    x=torch.randn(16, 3, 256, 256)
    resnet18=torchvision.models.resnet18(pretrained=True)
    # 删除所有训练变量等
    resnet18.eval()
    model=torch.nn.Sequential(
        resnet18.conv1, 
        resnet18.bn1
  )
    f1=model.forward(x)
    fused=fuse(model[0], model[1])
    f2=fused.forward(x)
    d=(f1 - f2).mean().item()
    print("error:", d)
 
3.6.1　图优化框架分析
实现代码如下所示：
def @relay_add_one(%x : Tensor((10,), f32)) {
    call_destination_passing @te_add_one(%x,  out=%b) 
} 

def @te_add_one(%a: NDArray, %b: NDArray) {
    var %n
    %A = decl_buffer(shape=[%n], src=%a)
    %B = decl_buffer(shape=[%n], src=%b) 
    // body ir contents need to be evolved
    for %i = 0 to 10 [data_par] {
        %B[%i] = %A[%i] + 1.0 
    }
}
3.7.2　算子（操作符）融合方案及示例
实现代码如下所示：
namespace transform {
Pass FuseOps(int fuse_opt_level) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = (Function f, IRModule m, PassContext pc) {
        int opt_level = fuse_opt_level == -1 ? pc->opt_level : fuse_opt_level;
        auto max_fuse_depth = pc->GetConfig("relay.FuseOps.max_depth", Integer(kMaxFusedOps));
        return Downcast<Function>(FuseOps(f, opt_level, max_fuse_depth.value(), m));
      };
  return CreateFunctionPass(pass_func, 1, "FuseOps", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.FuseOps").set_body_typed(FuseOps);
FuseOps依赖InferType的Pass，用PassContext获取配置信息，注册Pass的Python API。
构建DAG流程，先如下定位：
IndexedForwardGraph IndexedForwardGraph::Create(support::Arena* arena, const Expr& body) {
  return Creator(arena).Prepare(body);
}
// Creator类核，实现代码如下所示：
// 数据流后支配树的创建者
class IndexedForwardGraph::Creator : private ExprVisitor {
 public:
  explicit Creator(support::Arena* arena) : arena_(arena) {}
  IndexedForwardGraph Prepare(const Expr& body) {
    this->Update(body, nullptr, kOpaque);
    this->VisitExpr(body);
    return std::move(graph_);
  }
// 省略了成员变量函数
  .....
}
VisitExpr调用IndexedforwardGraph的VisitExpr_，基于深度优先遍历Relay，构造DAG图。先调用Update函数，再调用VisitExpr，最后插入根节点，初始化DAG。
DFS使用Update成员函数构建边，获得后序搜索树，实现代码如下所示： 
// 更新存储在节点上的信息
  void Update(const Expr& node, IndexedForwardGraph::Node* parent, OpPatternKind pattern) {
    const tvm::Object* key = node.get();
    IndexedForwardGraph::Node* current;
    auto it = graph_.node_map.find(key);
    if (it != graph_.node_map.end()) {
      current = it->second;
    } else {
      current = arena_->make<IndexedForwardGraph::Node>();
      graph_.node_map[key] = current;
    }
    if (parent != nullptr) {
      auto* link = arena_->make<LinkNode<IndexedForwardGraph::Edge> >();
      link->value.node = parent;
      link->value.pattern = pattern;
      current->outputs.Push(link);
    } else {
      current->extern_ref = true;
    }
  }
在tvm/include/tvm/relay/op_attr_types.h中的OpPatternKind pattern，表示节点与边的Operator运算类型。输出存储节点输入边，支持后序支配树计算LCA。搜索树DAG支持支配树算法，生成后序支配树。实现代码如下所示：
/*! \ 用于图融合的简单算子模式 */
enum OpPatternKind {
  // Elementwise operation
  kElemWise = 0,
  // 广播算子，始终可以将输出轴按顺序映射到输入
  // 例如：code:`out[i, ax1, j, ax2] = input[i, j]`.
  // 注意，轴需要按顺序排列，以便转置不是bcast算子
  kBroadcast = 1,
  // 内射映射算子，可以将输出轴内射映射到单个输入轴
  // 所有内射映射算子仍然可以安全地融合到内射映射算子和约化算子
  kInjective = 2,
  // 交互还原算子
  kCommReduce = 3,
  // 复杂算子，仍然可以将elemwise操作融合到输出中
  // 但无法链接另一个复杂的op
  kOutEWiseFusable = 4,
  // 元组节点的模式。可以融合到后续的内射映射操作中，但经过特殊处理
  kTuple = 7,
  // 不透明操作，无法融合任何内容
  kOpaque = 8
};
用IndexedForwardGraph::Creator改写visitExpr_，以便支持FunctionNode，ConstantNode, CallNode, TuppleNode等节点类型。CallNode的visitExpr_应用程序如下所示：
void VisitExpr_(const CallNode* call) final {
    ICHECK(graph_.node_map.count(call));
    Node* node = graph_.node_map.at(call);
    static auto fpattern = Op::GetAttrMap<TOpPattern>("TOpPattern");
    // 配置这个调用的模式
    // 如果调用一个算子，应该用标记带注释的图案
    // 如果图案没有注释，将默认为不透明
// 最后，如果算子位置不是调用节点，将需要调用Update，因为可能是任意表
式。
    OpPatternKind op_pattern = kOpaque;
    if (const OpNode* opnode = call->op.as<OpNode>()) {
      auto op = GetRef<Op>(opnode);
      if (IsDynamic(call->checked_type()) && IsDataDependent(call)) {
        // output of a shape func can't be fed to a data-dependent shape func
        op_pattern = kOpaque;
      } else {
        op_pattern = static_cast<OpPatternKind>(fpattern[op]);
      }
    } else {
      this->Update(call->op, node, kOpaque);
    }
    node->pattern = op_pattern;
    this->Update(call->op, nullptr, kOpaque);
    const auto* rtype = call->checked_type().as<TensorTypeNode>();
    // 将分析传递到引用的所有子级。
    for (size_t i = 0; i < call->args.size(); ++i) {
      const auto* arg_type = call->args[i]->checked_type().as<TensorTypeNode>();
      // 检查具体结果类型是否与参数类型相同
      OpPatternKind edge_pattern = op_pattern;
      if (edge_pattern == kBroadcast && arg_type != nullptr && rtype != nullptr &&attr_equal_(rtype->shape, arg_type->shape)) {
        edge_pattern = kElemWise;
      }
      this->Update(call->args[i], node, edge_pattern);
    }
    ExprVisitor::VisitExpr_(call);
    this->AddNode(call);
  }
VisitExpr_从叶节点开始，先进行深度优先搜索，再在输入DAG图中执行后序遍历。常量节点没有叶节点，ConstantNode的VisitExpr_不用递归调用。
实现CallNode的VisitExpr。先在DAG中加入输入，再遍历Edge，然后更新DAG。在ExprVisitor中，定义CallNode函数，实现代码如下所示：
void ExprVisitor::VisitExpr_(const CallNode* op) {
  this->VisitSpan(op->span);
  this->VisitExpr(op->op);
  for (auto ty_arg : op->type_args) {
    this->VisitType(ty_arg);
  }
  for (auto arg : op->args) {
    this->VisitExpr(arg);
  }
}
ExprVisitor派生IndexForwardGraph，this指向IndexForwardGraph，先调用VisitExpr_虚函数，然后实现relay树递归遍历。
2. 构造后序支配树
基于Relay后支配树创立DFS后序遍历DAG。现在来看看如何建立后序支配树，支配树的构建由DominatorTree类的PostDom成员函数来完成。最终的节点是后序遍历Relay支配树的根节点，先从根节点开始，再搜索相连节点的LCA，也是后序支配点。实现代码如下所示：
DominatorTree DominatorTree::PostDom(support::Arena* arena, const IndexedForwardGraph& graph) {
  DominatorTree tree;
  tree.nodes.resize(graph.post_dfs_order.size(), nullptr);
// 逆拓扑排序
  for (size_t i = graph.post_dfs_order.size(); i != 0; --i) {
    size_t index = i - 1;
    tree.nodes[index] = tree.GetNode(arena, graph.post_dfs_order[index]);
  }
  return tree;
}
用GetNode找到支配点与后支配树。实现代码如下所示：
/*! 	
初始化根节点，求各LCA支配点。用LeastCommonAncestor，计算所有LCA，程序如下所示：
/*!
    * \简单查找节点列表中最不常见的祖先
    * \param nodes the nodes
    * \param edge_pattern
    * \所有父对象的组合边缘模式，返回所有节点中最不常见的祖先
    */
Node* LeastCommonAncestor(const LinkedList<IndexedForwardGraph::Edge>& input_nodes, OpPatternKind* edge_pattern) {
    auto link = input_nodes.head;
    if (link == nullptr) {
      return nullptr;
    }
    auto get_node = [&](const IndexedForwardGraph::Edge& edge) {
      size_t oindex = edge.node->index;
      ICHECK_LT(oindex, nodes.size());
      Node* onode = nodes[oindex];
      ICHECK(onode != nullptr);
      return onode;
    };
    Node* parent = get_node(link->value);
    *edge_pattern = CombinePattern(*edge_pattern, link->value.pattern);
    link = link->next;
    for (; link != nullptr; link = link->next) {
      parent = LeastCommonAncestor(parent, get_node(link->value), edge_pattern);
      *edge_pattern = CombinePattern(*edge_pattern, link->value.pattern);
    }
    return parent;
  }
求开始2个节点的LCA，这是parent，遍历所有节点，得到LCA。在DAG中计算LCA程序实现如下所示：
/*!
* \简单查找节点列表中最不常见的祖先
  * \param lhs The left node.
  * \param rhs The right node.
  * \param edge_pattern
  * \组合所有父对象的边缘模式
  * \返回两者中最不常见的祖先
  */
static Node* LeastCommonAncestor(Node* lhs, Node* rhs, OpPatternKind* edge_pattern) {
    while (lhs != rhs) {
      if (lhs == nullptr) return nullptr;
      if (rhs == nullptr) return nullptr;
      if (lhs->depth < rhs->depth) {
        edge_pattern[0] = CombinePattern(edge_pattern[0], rhs->pattern);
        rhs = rhs->parent;
      } else if (rhs->depth < lhs->depth) {
        edge_pattern[0] = CombinePattern(edge_pattern[0], lhs->pattern);
        lhs = lhs->parent;
      } else {
        edge_pattern[0] = CombinePattern(edge_pattern[0], lhs->pattern);
        edge_pattern[0] = CombinePattern(edge_pattern[0], rhs->pattern);
        lhs = lhs->parent;
        rhs = rhs->parent;
      }
    }
    return lhs;
  }
深度不同的2个节点，沿父节点往上爬，一旦深度一致，就是LCA节点。支配点pattern的最大值，就是LCA的pattern。计算pattern最小值到最大值，如kElemWise=0, kInjective=2, 前者可向下融合进KInjective。
3. 执行算子融合
实现代码如下所示：
std::vector<GraphPartitioner::Group*> GraphPartitioner::Partition(
    const IndexedForwardGraph& graph) {
  this->InitGroups(graph);
  if (opt_level_ == 0) return std::move(groups_);
  //获取后支配树
  auto post_dom_tree = DominatorTree::PostDom(arena_, graph);
  //运行融合算法
  for (int phase = 0; phase < 3; ++phase) {
    this->RunFuse(graph, post_dom_tree, phase);
  }
  return std::move(groups_);
}
Group结构体与InitGroups的程序实现如下所示：
struct Group {
    /*! \简要介绍在union中查找数据结构中的父级* */
    Group* parent{nullptr};
    /*! \简述团队模式 */
    OpPatternKind pattern;
    /*! \简要引用根节点*/
    const tvm::Object* root_ref{nullptr};
    /*!
     * \anchor节点的简要参考，仅当模式为kOutEWiseFusable时，此字段才不是nullptr
     */
    const tvm::Object* anchor_ref{nullptr};
    /*!
     * \简要查找组根节点，执行路径压缩
     * \返回根类型节点
     */
    Group* FindRoot() {
      // fast path
      if (this->parent == nullptr) return this;
      //具有路径压缩的慢路径
      Group* root = this;
      while (root->parent != nullptr) {
        root = root->parent;
      }
      for (Group* p = this; p != root;) {
        Group* parent = p->parent;
        p->parent = root;
        p = parent;
      }
      return root;
    }

    /*!
     * \简要说明属于此组的节点数
     */
    uint32_t num_nodes{1};
  };

//初始化组
void InitGroups(const IndexedForwardGraph& graph) {
    groups_.resize(graph.post_dfs_order.size());
    for (size_t nid = 0; nid < groups_.size(); ++nid) {
      const auto* graph_node = graph.post_dfs_order[nid];
      auto* group_node = arena_->make<Group>();
      group_node->pattern = graph_node->pattern;
      group_node->root_ref = graph_node->ref;
      //如有必要，设置anchor参考
      if (group_node->pattern == kOutEWiseFusable) {
        group_node->anchor_ref = graph_node->ref;
      }
      groups_[nid] = group_node;
    }
  }
实现代码如下所示：
// 实现融合算法
void RunFuse(const IndexedForwardGraph& graph, const DominatorTree& post_dom_tree, int phase) {
    // 从顶部搜索起, 初始态groups_, 相当于执行IndexedForwardGraph。
    for (size_t nid = 0; nid < groups_.size(); ++nid) {
      // 执行已指定当前节点的组
      auto* graph_node = graph.post_dfs_order[nid];
      auto* dom_node = post_dom_tree.nodes[nid];
      Group* group_node = groups_[nid];
      ICHECK(group_node != nullptr);
      // 不透明节点无操作
      if (group_node->pattern == kOpaque) continue;
      // 如果当前节点没有支配者，无须执行任何操作
      if (dom_node->parent == nullptr) continue;
      ICHECK(!graph_node->extern_ref);
      size_t dom_parent_gindex = dom_node->parent->gnode->index;
	  
	  // 多于某个数据的max_fuse_depth_节点, 就不进行融合
      if (CountFusedNodesWithNewChild(graph_node, dom_node->parent->gnode) > max_fuse_depth_)
        continue;

      if (phase == 2) {
        // 将内射映射运算融合到中间元组中（如果有）
        if (group_node->pattern > kInjective) continue;
        Group* dom_parent_group = groups_[dom_parent_gindex];
        Group* dom_root_group = dom_parent_group->FindRoot();
        // 如果dom节点组有一个元组作为根，不会将元组字段融合
        if (dom_root_group->pattern == kTuple) continue;
        if (dom_parent_group->pattern == kTuple && dom_root_group->pattern <= kInjective) {
          // 现在知道元组已经融合到后续的内射映射运算中
          auto fcond = [](OpPatternKind kind, bool is_sink) { return kind <= kInjective; };
          // dom_root_group也可以是元组，如初始层
          // 需要检查路径以避免融合两个中间元组
          if (CheckPath(graph_node, dom_node->parent->gnode, fcond)) {
            CommitFuse(graph_node, dom_node->parent->gnode);
          }
        }
        continue;
      }
      // 若融合了当前节点与父节点, 跳过当前节点, 防止重复融合
      if (groups_[dom_parent_gindex] != nullptr &&
          group_node->FindRoot() == groups_[dom_parent_gindex]->FindRoot()) {
        continue;
      }
      // 暂时不要融合到元组中
      if (groups_[dom_parent_gindex]->pattern == kTuple) continue;
      // 尝试将当前节点融合到其后控制器
      if (group_node->pattern == kOutEWiseFusable) {
        if (phase != 0) continue;
        // Path for OutEWiseFusable: conv2dConv2d
        // 检查支配关系是否为elemwise
        if (dom_node->parent != nullptr && dom_node->pattern == kElemWise) {
          ICHECK(dom_node->parent->gnode != nullptr);
          // 如果所有中间操作仍在广播中，可以执行融合
          auto fcond = [](OpPatternKind kind, bool is_sink) { return kind <= kBroadcast; };
          if (CheckPath(graph_node, dom_node->parent->gnode, fcond)) {
            CommitFuse(graph_node, dom_node->parent->gnode);
          }
        }
      } else if (group_node->pattern <= kBroadcast) {
        // 先决条件：只能融合到内射映射或还原的父级
        if (dom_node->parent != nullptr &&
            (dom_node->pattern <= kInjective || dom_node->pattern == kCommReduce)) {
          // 检查所有中间操作是否仍在广播
          // 最终的终端节点已经可以融合到一个OutEWiseFusable组
          auto fcond = [](OpPatternKind kind, bool is_sink) {
            if (!is_sink) {
              // 并行分支上的Elemwise、广播和内射映射算子可以融合到Elemwise/广播anchor。
              return kind <= kInjective;
            } else {
              return (kind <= kBroadcast || kind == kCommReduce || kind == kInjective || kind == kOutEWiseFusable);
            }
          };
          if (CheckPath(graph_node, dom_node->parent->gnode, fcond)) {
            CommitFuse(graph_node, dom_node->parent->gnode);
          }
        }
      } else if (group_node->pattern == kInjective || group_node->pattern == kTuple) {
        // 将内射映射融合推迟到第二阶段
        // 所以conv2dConv2d总是完成融合
        if (phase != 1) continue;
        // 检查所有路径是否都是内射映射的
        auto fcond = [](OpPatternKind kind, bool is_sink) { return kind <= kInjective; };
        if (CheckPath(graph_node, dom_node->parent->gnode, fcond)) {
          CommitFuse(graph_node, dom_node->parent->gnode);
        }
      } else {
        // 什么都不做
        ICHECK(group_node->pattern == kCommReduce);
      }
    }
  }
};
实现代码如下所示：
/*!
* \简要检查src和接收器之间的所有节点和边缘模式是否满足fcond
*
* 未检查src
*
* \param src 源节点
* \param sink 接收终止节点
* \param fcond 要检查的条件
* \tparam F条件函数的RAM，带有签名
* \注意sink必须是src的后支配者
*/
template <typename F>
  bool CheckPath(IndexedForwardGraph::Node* src, IndexedForwardGraph::Node* sink, F fcond) {
    ICHECK(!src->extern_ref);
    visited_.clear();
    ICHECK(src != sink);
    for (auto link = src->outputs.head; link != nullptr; link = link->next) {
      if (!CheckPath_(link->value.node, sink, fcond)) return false;
    }
    return true;
  }
若支配树节点能融合，使用CommitFuse实现融合运算，实现代码如下所示：
/*!
    * \简要提交融合操作
    * \param src 源节点
    * \param sink 接收终止节点
    * \注意sink必须是src的后支配者
    */
void CommitFuse(IndexedForwardGraph::Node* src, IndexedForwardGraph::Node*sink) {
    Group* target = groups_[sink->index];
    visited_.clear();
    ICHECK(src != sink);
    CommitFuse_(src, sink, target);
  }
指定融合节点，添加Group* target指针，通过使用CommitFuse_方法实现融合。实现代码如下所示：
// ommitFuse内部实施
void CommitFuse_(IndexedForwardGraph::Node* src, IndexedForwardGraph::Node* sink, Group* target) {
    if (src == sink) return;
    if (visited_.count(src)) return;
    visited_.insert(src);
    Group* gnode = groups_[src->index];
    ICHECK(gnode != nullptr);
    // 如果可能，将当前组合并到父组
    MergeFromTo(gnode, target);
    for (auto link = src->outputs.head; link != nullptr; link = link->next) {
      CommitFuse_(link->value.node, sink, target);
    }
  }
执行MergeFromTo(gnode, target) 节点融合，实现代码如下所示：
/*!
* \将子组简要合并到父组
* \param child 子组
* \param parent 父组
*/
  void MergeFromTo(Group* child, Group* parent) {
    child = child->FindRoot();
    parent = parent->FindRoot();
    if (child == parent) return;
    // 更新父组的节点数
    parent->num_nodes += child->num_nodes;
    child->parent = parent;
    // 更新anchor参数和模式
    if (child->anchor_ref != nullptr) {
      ICHECK(parent->anchor_ref == nullptr);
      parent->anchor_ref = child->anchor_ref;
      parent->pattern = CombinePattern(child->pattern, parent->pattern);
    }
  }
child->FindRoot()函数，搜索目前节点的父节点。如融合A-B-C，B的parent是C，A的parent是C。用前4行运算，target或root表示中间节点的parent。
实现代码如下所示：
// Run the transform
  Expr Transform(const Expr& body, int fuse_opt_level, size_t max_fuse_depth) {
    // 设置组映射
    auto graph = IndexedForwardGraph::Create(&arena_, body);
    auto groups = GraphPartitioner(&arena_, fuse_opt_level, max_fuse_depth).Partition(graph);
    for (size_t nid = 0; nid < graph.post_dfs_order.size(); ++nid) {
      ICHECK(graph.post_dfs_order[nid]->ref != nullptr);
      gmap_[graph.post_dfs_order[nid]->ref] = groups[nid];
    }
    // 以下行可用于调试
    // this->DebugDumpGroup(body);
    return this->Mutate(body);
  }
将可融合列节点，生成Function Expr返回，进行算子融合Pass。总之，本示例解析了TVM Pass支配树与算子融合实施方案。
3.8　控制流与优化器
3.8.1　控制流
Tensorflow中条件判断实现Ccond(pre, fn1, fn2) 条件判断实现的伪代码如下： 
# 构建真实分支图形
context_t=CondContext(pred, branch=1)
res_t=context_t.Call(fn1)
# 构建假分支图
context_f=CondContext(pred, branch=0)
res_f=context_f.Call(fn2)
# 添加输出合并节点
merges=[Merge([f, t]) for (f, t) in zip(ref_f, res_t)]
return merges 
对于循环语句，tensorflow中使用一下伪代码来完成如下所示：
while_context=WhileContext()
while_context.Enter()
# 为每个循环变量添加输入节点
enter_vars=[Enter(x, frame_name) for x in loop_vars]
# 添加合并节点，输入将在稍后更新
merge_vars=[Merge([x, xx]) for x in enter_vars]
# 构建循环pred子图
pred_result=pred(*merge_vars)
# 添加Switch节点
switch_vars=[Switch(x, pred_result) for x in merge_vars]
# 构建循环体子图
body_result=body(*[x[1] for x in switch_vars])
# 添加NextIteration节点
Next_vars=[NextIteration(x) for x in body_result]
# 形成循环的循环
for m, v in zip(merge_vars, next_vars):'
   m.op._update_input(1, v)
# 添加退出节点
exit_vars=[Exit(x[0]) for x in switch_vars]
while_context.Exit()
return exit_vars
3.10　多功能张量加速器VTA
3.10.3　VTA示例
显示了C ++中定义的VTA模块之一的定义：
void fetch(
  uint32_t insn_count,
  volatile insn_T *insns,
  hls::stream<insn_T> &load_queue,
  hls::stream<insn_T> &gemm_queue,
  hls::stream<insn_T> &store_queue) {
#pragma HLS INTERFACE s_axilite port = insn_count bundle = CONTROL_BUS
#pragma HLS INTERFACE m_axi port = insns offset = slave bundle = ins_port
#pragma HLS INTERFACE axis port = load_queue
#pragma HLS INTERFACE axis port = gemm_queue
#pragma HLS INTERFACE axis port = store_queue
#pragma HLS INTERFACE s_axilite port = return bundle = CONTROL_BUS

  INSN_DECODE: for (int pc = 0; pc < insn_count; pc++) {
#pragma HLS PIPELINE II = 1
    // 读取指令字段
    insn_T insn = insns[pc];
    // 进行部分解码
    opcode_T opcode = insn.range(VTA_INSN_MEM_0_1, VTA_INSN_MEM_0_0);
    memop_id_T memory_type = insn.range(VTA_INSN_MEM_5_1, VTA_INSN_MEM_5_0);
    // 推送到适当的指令队列
    if (opcode == VTA_OPCODE_STORE) {
      store_queue.write(insn);
    } else if (opcode == VTA_OPCODE_LOAD &&
        (memory_type == VTA_MEM_ID_INP || memory_type == VTA_MEM_ID_WGT)) {
      load_queue.write(insn);
    } else {
      gemm_queue.write(insn);
    }
  }
}
3.11　TVM代码库结构与示例
3.11.2　Tensor添加示例
这是用TVM降级（Lowering）下译（Lowering）API实现向量加法示例。实现代码如下所示：
n = 1024
A = tvm.te.placeholder((n,), name='A')
B = tvm.te.placeholder((n,), name='B')
C = tvm.te.compute(A.shape, lambda i: A[i] + B[i], name="C")
其中A, B, C表示python/tvm/te/tensor.py中tvm.tensor.Tensor。C++支持Python 张量，在include/tvm/te/tensor.h与src/te/tensor.cc中执行。TVM 中Python可看作同名C++句柄。
通过TVM_REGISTER_*宏，按照PackedFunc方式开放C++。TVM 通过PackedFun实现 C++ 和 Python 间互操作性，使得从 C++ 代码库调用 Python 函数变得容易。
张量包括运算关联，在python/tvm/te/tensor.py，include/tvm/te/operation.h与src/tvm/te/operation中定义。张量是运算对象的输出，而每个运算都有input_tensors()方法，返回输入列表张量。
将张量传到python/tvm/te/schedule.py中的 tvm.te.create_schedule函数。实现代码如下所示：
s = tvm.te.create_schedule(C.op)
// 函数映射到include/tvm/schedule.h中
inline Schedule create_schedule(Array<Operation> ops) {
  return Schedule(ops);
}
实现代码如下所示：
target = "cuda"
fadd = tvm.build(s, [A, B, C], target)
下译（Lowering）代码的实现：
def lower(sch,
          args,
          name="default_function",
          binds=None,
          simple_mode=False):
   ...
   bounds = schedule.InferBound(sch)
   stmt = schedule.ScheduleOps(sch, bounds)
   ...
实现代码如下所示：
...
stmt = ir_pass.VectorizeLoop(stmt) 
...
stmt = ir_pass.UnrollLoop(
    stmt,
    cfg.auto_unroll_max_step,
    cfg.auto_unroll_max_depth,
    cfg.auto_unroll_max_extent,
    cfg.unroll_explicit)
...
代码如下所示：
TVM_REGISTER_GLOBAL("codegenCodeGen.build_cuda") 
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = BuildCUDA(args[0]);
  });
实现代码如下所示：
dev = tvm.device(target, 0)
a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
fadd(a, b, c)
output = c.numpy()
实现代码如下所示： 
PackedFunc CUDAModuleNode::GetFunction(
      const std::string& name,
      const std::shared_ptr<ModuleNode>& sptr_to_self) {
  auto it = fmap_.find(name);
  const FunctionInfo& info = it->second;
  CUDAWrappedFunc f;
  f.Init(this, sptr_to_self, name, info.arg_types.size(), info.launch_param_tags);
  return PackFuncVoidAddr(f, info.arg_types);
}
实现代码如下所示： 
class CUDAWrappedFunc {
 public:
  void Init(...)
  ...
  void operator()(TVMArgs args,
                  TVMRetValue* rv,
                  void** void_args) const {
    int device_id;
    CUDA_CALL(cudaGetDevice(&device_id));
    if (fcache_[device_id] == nullptr) {
      fcache_[device_id] = m_->GetFunc(device_id, func_name_);
    }
    CUstream strm = static_cast<CUstream>(CUDAThreadEntry::ThreadLocal()->stream);
    ThreadWorkLoad wl = launch_param_config_.Extract(args);
    CUresult result = cuLaunchKernel(
        fcache_[device_id],
        wl.grid_dim(0),
        wl.grid_dim(1),
        wl.grid_dim(2),
        wl.block_dim(0),
        wl.block_dim(1),
        wl.block_dim(2),
        0, strm, void_args, 0);
  }
};
这样就完成了TVM 如何编译和执行函数的流程。

 
第4章　TVM量化技术
4.1　 TVM量化概述
实际上，会在TVM伪量化中组合两种误差进行消化。执行代码如下所示：
@_op.register_compute("relay.op.annotation.simulated_quantize")
def simulated_quantize_compute(attrs, inputs, out_type):
    // 模拟量化编译器
    assert len(inputs) == 4
    assert attrs.sign
    assert attrs.rounding == "round"
    data, scale, clip_min, clip_max=inputs
    if attrs.kind == QAnnotateKind.IDENTITY:
        return [topi.identity(data)]    
    scaled_data=topi.divide(data, scale)
    # 模拟饱和误差
    clipped_data=topi.maximum(topi.minimum(scaled_data, clip_max), clip_min)
    # 模拟舍入误差
    round_data=topi.round(clipped_data)
    # 恢复data
    rdata=topi.multiply(round_data, scale)
    return [rdata]
4.2　 int8量化与tvm执行
4.2.4　实现int8量化
实现一个量化示例。
# load mxnet, onnx等前端模型
sym, _=relay.frontend.from_mxnet(sym, {'data': data_shape})
# 随机产生test模型参数, 除非有现成训练好的模型参数
sym, params=tvm.relay.testing.create_workload(sym)
# 模型量化
with relay.quantize.qconfig(skip_k_conv=0, round_for_shift=True):
    sym=relay.quantize.quantize(sym, params)
# tvm默认resnet卷积优化配置, 包括内核数量
# 用新的卷积结构, 进行auto tuning优化。若用现存的卷积优化, 速度更快。
# load编译模型的优化算子
with autotvm.apply_history_best(log_file): 
    print("Compile...")
    with relay.build_config(opt_level=3):
        graph, lib, params=relay.build_module.build(
            net, target=target, params=params)
    # load 参数并运行
    ctx=tvm.context(str(target), 0)
    module=runtime.create(graph, lib, ctx)
    data_tvm=tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
    module.set_input('data', data_tvm)
    module.set_input(**params)
    # module.set_input(**{k:tvm.nd.array(v, ctx) for k, v in params.items()})
    module.run()
    # forward时间测试
    e=module.module.time_evaluator("run", ctx, number=2000, repeat=3)
    t=module(data_tvm).results
    t=np.array(t) * 1000
print('{} (batch={}): {} ms'.format(name, batch, t.mean()))

4.3　低精度训练与推理
下面是具体量化方法，包括线性量化，对数线性量化及双曲正切量化三个模块，实现代码如下所示：
# 线性量化
def linear_quantize(input, sf, bits):
    assert bits >= 1, bits
    # 一bit
    if bits == 1:
        return torch.sign(input) - 1  
    delta=math.pow(2.0, -sf)# 小数位位宽量化精度
    bound=math.pow(2.0, bits-1)
    min_val=- bound    # 上限值
    max_val=bound - 1  # 下限值
    rounded=torch.floor(input/delta+0.5) # 扩大后取整
    clipped_value=torch.clamp(rounded, min_val, max_val) * delta # 再缩回
    return clipped_value
# 对数线性量化
def log_linear_quantize(input, sf, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input), 0.0, 0.0
    s=torch.sign(input) # 正负号
    input0=torch.log(torch.abs(input)+1e-20)  # 求对数获取比特位
    v=linear_quantize(input0, sf, bits)       # 对比特位进行线性量化
    v=torch.exp(v) * s                        # 指数回原数
    return v
# 双曲正切量化
def tanh_quantize(input, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input)
    input=torch.tanh(input)         # 双曲正切映射 [-1, 1]
    input_rescale=(input+1.0)/2     # 映射到 [0, 1]
    n=math.pow(2.0, bits) - 1       # 固定比特位放大系数
    v=torch.floor(input_rescale * n+0.5)/n   # 放大后取整
    v=2 * v - 1 # [-1, 1]                    # 放回原来的范围
    v=0.5 * torch.log((1+v)/(1 - v))         # 反双曲正切回原数 arctanh
    return v
4.6　熵校准示例
通过示例来进行熵校准操作：
Input：FP32直方图，包括2048个bins单元：bin[0]，…，bin[2047]
实现代码如下所示：
for i in range( 128, 2048): 
reference_distribution_P=[bin[0], ..., bin[i-1]] // take first ‘ i ‘ bins from H 
outliers_count=sum( bin[i], bin[i+1], …, bin[2047]) 
reference_distribution_P[i-1] += outliers_count 
P/=sum(P) // normalize distribution P 
candidate_distribution_Q=quantize [bin[0], …, bin[i-1]] into 128 levels // explained later 
expand candidate_distribution_Q to ‘ i ’ bins // explained later 
Q/=sum(Q) // normalize 
distribution Q divergence[i]=KL_divergence( reference_distribution_P, 
 candidate_distribution_Q) 
End For 
找到divergence[m]最小的索引“m”：
threshold=( m+0.5) * ( width of a bin)
以下是实现INT8 卷积内核的伪代码：
// I8 输入张量：I8_input, I8_weights, I8 output tensors: I8_output 
// F32 bias (原始bias来自F32模型) 
// F32 缩放因子：input_scale, output_scale, weights_scale[K] 
I32_gemm_out=I8_input * I8_weights // Compute INT8 GEMM (DP4A) 
F32_gemm_out=(float)
I32_gemm_out // Cast I32 GEMM output to F32 float 
// 这一点上，包括input_scale * weights_scale[K]缩放的F32_gemm_out
// 使scale=“output_scale”，在int8中保存，再进行scale:
// 此乘法采用NCHW格式，可在F32中，使用*_gemm_out数组完成
For i in 0, ... K-1: 
rescaled_F32_gemm_out[:, i, :, :]=F32_gemm_out[:, i, :, :] * [output_scale/(input_scale * weights_scale[i])] 
// 增加bias, 执行“output_ratio”缩放，再缩放F32 bias: 
rescaled_F32_gemm_out _with_bias=rescaled_F32_gemm_out+output_scale * bias // 
Perform ReLU (in F32) 
F32_result=ReLU(rescaled_F32_gemm_out _with_bias) 
// 转换为INT8, 保存为全局
I8_output=Saturate( Round_to_nearest_integer( F32_result))
4.7　 TVM量化流程
4.7.4　量化处理硬件说明
代码实现如下所示：
desc=Hardware()
desc['add'].append(OpDesc(in_dtypes=['int32', 'int32'], out_dtypes=['int32']))
desc['add'].append(OpDesc(in_dtypes=['float32', 'float32'], out_dtypes=['float32']))
desc['nn.conv2dConv2d'].append(OpDesc(in_dtypes=['int16', 'int16'], out_dtypes=['int32']))
desc['nn.conv2dConv2d'].append(OpDesc(in_dtypes=['int8', 'int8'], out_dtypes=['int32']))
desc['nn.global_avg_pool2d'].append(OpDesc(in_dtypes=['float32', 'float32'], out_dtypes=['float32']))
4.7.5　阈值估计方案
实现代码如下所示：
@register_fthreshold_rectify('add')
def threshold_rectify_for_add(in_bits, out_bits, in_tholds, out_tholds):
   # choose scale of the one with maximum threshold
   idx=np.argmax(in_tholds)
   unified_scale=in_tholds[idx]/(2**(in_bits[idx] - sign_bit))
   # adjust thresholds according to the unified scale
   ...
4.7.6　模拟量化误差
将simulated_quantize在每条边上，插入一个算子，试图模拟这些错误。具体实现代码如下所示：
def simulated_quantize(data, in_scale, out_scale, clip_min, clip_max, in_dtype, out_dtype):
    if in_dtype == 'float32' and out_dtype == 'float32':
        # no need to quantize
        return data    
    # simulated overflow error    
    data=data/in_scale
    data=topi.cast(data, in_dtype)
    data=data * in_scale
    scaled_data=data/out_scale
    # simulate saturated error
    clipped_data=topi.clip(scaled_data, clip_min, clip_max)
    # simulate round error
    rounded_data=topi.cast(topi.round(scaled_data), out_dtype)
    out=rounded_data * out_scale
    return out
用位与阈值计算参数。如何通过位和阈值，计算这些参数呢？out_scale、clip_min、clip_max 是非常严格的，对于in_scale、in_dtype、out_dtype，需要做额外推理。代码实现如下所示：
integer_range=2**(bit - sign_bit)
out_scale=threshold/integer_range
clip_min =- (integer_range - 1)
clip_max = integer_range - 1
对于in_scale、in_dtype、out_dtype，需要做额外推理。
3. 接口预定义描述
    先调用库文件，再进行预定义描述。代码实现如下所示：
from tvm import hago
# 理想情况下，将对x86、arm、gpu和vta进行预定义描述
hardware=hago.create_sample_hardware()
strategy, sim_acc=hago.search_quantize_strategy(graph, hardware, dataset)
quantizer=hago.create_quantizer(graph, hardware, strategy)
simulated_graph=quantizer.simulate()
quantized_graph=quantizer.quantize()
4.8　TVM量化程序分析
实现代码如下所示：
def test_mul_rewrite():
    // mul的rhs不是常数的测试用例
    data=relay.var("data", shape=(1, 16, 64, 64))
    multiplier=relay.sigmoid(relay.var("data", shape=(1, 16, 1, 1)))
    conv=relay.nn.conv2dConv2d(data, relay.var("weight"), 
                           kernel_size=(3, 3), 
                           padding=(1, 1), 
                           channels=16)
    act=relay.nn.relu(data=conv)
    quantize_and_build(act * multiplier)
    pool=relay.nn.global_avg_pool2d(data=act)
    quantize_and_build(act * pool)
//入口函数：
def quantize_and_build(out):
    f=relay.Function(relay.analysis.free_vars(out), out)
    mod, params=testing.create_workload(f)
    with relay.quantize.qconfig(skip_conv_layers=[]):
        qmod=relay.quantize.quantize(mod, params)
        
    relay.build(qmod, "llvm", params=params)
    return qmod
//用relay.quantize.quantize函数主体
// 1:步骤1，优化
   mod=prerequisite_optimize(mod, params)
// 2: 步骤2，量化配置
   calibrate_pass=tvm.transform.module_pass(
        calibrate(dataset), opt_level=1, 
        name="QuantizeCalibrate")
    quant_passes=[partition(), 
                    annotate(), 
                    calibrate_pass]
    if not current_qconfig().do_simulation:
        quant_passes.append(realize())
    quant_passes.append(_transform.FoldConstant())
    quantize_seq=tvm.transform.Sequential(quant_passes)

    with tvm.transform.PassContext(opt_level=3, 
                                   required_pass=["QuantizeAnnotate", 
                                                  "QuantizeCalibrate", 
                                                  "QuantizeRealize"]):
   // 3:步骤3，推理图精度转
with quantize_context():
            mod=quantize_seq(mod)
// 4: 步骤3，实现推理
   q_cfg=current_qconfig()
    assert q_cfg.partition_conversions in ['disabled', 'enabled', 'fully_integral']
    if q_cfg.partition_conversions != 'disabled':
        quantized_dtypes={q_cfg.dtype_input, q_cfg.dtype_weight, q_cfg.dtype_activation}
        ensure_fully_integral=q_cfg.partition_conversions == 'fully_integral'
        return partition_conversions(mod, quantized_dtypes, ensure_fully_integral)

 

第5章　TVM 优化调度
5.1　 TVM 运行时系统
5.1.2　 PackedFunc编译与部署
这里提供C++ 加法与调用，实现代码如下所示：
#include <tvm/runtime/packed_func.h>

void MyAdd(TVMArgs args, TVMRetValue* rv) {
// 自动将参数转换为所需类型
  int a=args[0];
  int b=args[1];
  // 自动分派值返回到rv
  *rv=a+b;
}

void CallPacked() {
  PackedFunc myadd=PackedFunc(MyAdd);
  // get back 3
  int c=myadd(1, 2);
}
在上面的代码块中，定义了一个PackedFunc MyAdd。接受两个参数: args表示输入参数，rv表示返回值。该函数是类型擦除的，这意味着函数签名不限制传递的输入类型或返回的类型。在底层，当调用PackedFunc时，它将输入参数打包到栈上的TVMArgs，并通过TVMRetValue返回结果。
多亏了C++中的模板技巧，可以像调用普通函数一样调用PackedFunc。由于它的类型擦除特性，可以从Python等动态语言中调用PackedFunc，而不需要为创建的每个新类型函数添加额外的胶水代码。下面的例子在C++中注册了PackedFunc，并从Python中调用。实现代码如下所示：
// 在C++中注册全局packed函数// 在C++中注册全局packed压缩函数++
TVM_REGISTER_GLOBAL("myadd")
.set_body(MyAdd);
import tvm

myadd=tvm.get_global_func("myadd")
# prints 3
print(myadd(1, 2))
实现代码如下所示：
TVM_REGISTER_GLOBAL("callhello")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  PackedFunc f=args[0];
  f("hello world");
});
import tvm

def callback(msg):
  print(msg)
# convert to PackedFunc
f=tvm.convert(callback)
callhello=tvm.get_global_func("callhello")
# 打印hello world
callhello(f)
5.1.3　构建 PackedFunc模块
5.1.5　TVM 对象与编译器分析
实现代码如下所示：
class AttrVisitor {
public:
  virtual void Visit(const char* key, double* value)=0;
  virtual void Visit(const char* key, int64_t* value)=0;
  virtual void Visit(const char* key, uint64_t* value)=0;
  virtual void Visit(const char* key, int* value)=0;
  virtual void Visit(const char* key, bool* value)=0;
  virtual void Visit(const char* key, std::string* value)=0;
  virtual void Visit(const char* key, void** value)=0;
  virtual void Visit(const char* key, Type* value)=0;
  virtual void Visit(const char* key, ObjectRef* value)=0;
  // ...
};
class BaseAttrsNode : public Object {
public:
  virtual void VisitAttrs(AttrVisitor* v) {}
  // ...
};
每个对象子类都将覆盖此项访问成员。下面是TensorNode的一个示例实现。实现示例如下所示：
class TensorNode : public Object {
public:
  /*! \简述张量的形状 */
  Array<Expr> shape;
  /*! \张量内容中的简要数据类型*/
  Type dtype;
  /*! \简要说明源代码操作，可以是None*/
  Operation op;
  /*! \简要介绍源算子的输出索引*/
  int value_index{0};
  /*! \简要构造函数*/
  TensorNode() {}
  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("shape", &shape);
    v->Visit("dtype", &dtype);
    v->Visit("op", &op);
    v->Visit("value_index", &value_index);
  }
};
在上面的示例中，操作和数组<Expr>都是ObjectRef。VisitAttrs提供了一个反射API，访问对象的每个成员。可以使用此函数访问节点，递归序列化任何语言对象。允许用前端语言轻松获取对象的成员。例如，在下面的代码中，访问了TensorNode的op字段：
import tvm
from tvm import te

x=te.placeholder((3, 4), name="x")
# access the op field of TensorNode
print(x.op.name)
5.2.2　动态图实现示例
代码实现如下所示：
import numpy as np

class OP:
    def __init__(self):
        self.name=self.__class__.__name__
    def __call__(self, *args):
        self.input=args  # save for backward
        self.output=self.forward(*args)
        self.output.op=self
        return self.output
    def forward(self, *args):
        raise NotImplementedError()
    def backward(self, *args):
        raise NotImplementedError()
    def backward_native(self, grad):
        # 上层传递的梯度, 保存OP的output梯度。
        self.output.grad=grad   
        # 通过upper layer传递的grad，调用backward, 计算OP的输入梯度的error项
        input_grads=self.backward(grad)
        # input有多个, 如AddOP运算a+b, 输入有a与b两个
        # input可能是一个, 如e^a运算, 输入只有一个a
        # 将input_grads变成tuple
        if not isinstance(input_grads, tuple):
            input_grads=(input_grads, )
        # 断言: assert (condition[, "...error info..."])
        # 若条件不满足，AssertionError("...error info...")异常
        # input_grads与input一致。若判断不一致，抛出异常
        assert len(input_grads) == len(self.input), "梯度的数量与输入的数量不匹配"
        # 搜索输入的张量, 递归调用backard梯度
        for input_grad, ip in zip(input_grads, self.input):
            if isinstance(ip, Tensor):
                ip.backward(input_grad)
    # 判断item是否是张量, 若是返回item.data；否则返回item
    def get_data(self, item):
        if isinstance(item, Tensor):
            return item.data
        else:
            return item

# Add运算: a+b
class AddOP(OP):
    def __init__(self):
        super().__init__()
        
    def forward(self, a, b):
        return Tensor(self.get_data(a)+self.get_data(b))
    def backward(self, grad):
        # a+b=c运算，grad_a=1*grad_c =grad_c。同理grad_b=grad_c
        # grad是高级传输的梯度，即grad_c
        return grad, grad

# Sub运算: a - b
class SubOP(OP):
    def __init__(self):
        super().__init__()
    def forward(self, a, b):
        return Tensor(self.get_data(a) - self.get_data(b))
    def backward(self, grad):
        return grad, -1 * grad
    
# Mul运算: a * b
class MulOP(OP):
    def __init__(self):
        super().__init__()
    def forward(self, a, b):
        return Tensor(self.get_data(a) * self.get_data(b))
    def backward(self, grad):
        a, b=self.input
        return grad * self.get_data(b), grad * self.get_data(a)
    
# Div运算: a * b
class DivOP(OP):
    def __init__(self):
        super().__init__()
    def forward(self, a, b):
        return Tensor(self.get_data(a)/self.get_data(b))
    def backward(self, grad):
        a, b=self.input
        return grad/self.get_data(b), grad * self.get_data(a)/(self.get_data(b) ** 2) * (-1)

# Exp运算: e^a
class ExpOP(OP):
    def __init__(self):
        super().__init__()
    def forward(self, a):
        return Tensor(np.exp(self.get_data(a)))
    def backward(self, grad):
        a=self.input[0]
        return grad * np.exp(self.get_data(a))
    
# Log运算: loga
class LogOP(OP):
    def __init__(self):
        super().__init__()
    def forward(self, a):
        return Tensor(np.log(self.get_data(a)))
    def backward(self, grad):
        a=self.input[0]
        return grad/self.get_data(a)
    
# 矩阵乘法: a @ b
class MatMulOP(OP):
    def __init__(self):
        super().__init__()
    def forward(self, a, b):
        return Tensor(self.get_data(a) @ self.get_data(b))
    def backward(self, grad):
        a, b=self.input
        return grad @ self.get_data(b).T, self.get_data(a).T @ grad

# SumOP: 对Tensor/Placeholder求和, 将矩阵相加
class SumOP(OP):
    def __init__(self):
        super().__init__()
    def forward(self, a):
        return Tensor(np.sum(self.get_data(a)))
    def backward(self, grad):
        a=self.input[0]  # SumOP求和输入input只有一个
        # SumOP输入梯度误差等于SumOP输出梯度误差
        # self.get_data(a)返回numpy类型数据
        return np.full_like(self.get_data(a), grad)
    
# MeanOP: 对Tensor/Placeholder平均, 可将矩阵相加, 除以总个数
class MeanOP(OP):
    def __init__(self):
        super().__init__()
    def forward(self, a):
        return Tensor(np.mean(self.get_data(a)))
    def backward(self, grad):
        a=self.input[0]  # SumOP求和输入input只有一个
        #对SumOP输入梯度误差等于对SumOP输出梯度误差
        # self.get_data(a)返回numpy数据
        d=self.get_data(a)
        return np.full_like(d, grad/d.size)
    
# 自定义张量: 包括op存储运算，data存储数据，grad存储梯度 
class Tensor:
    def __init__(self, data, op=None):
        self.data=data
        self.grad=None
        self.op=op
    def __radd__(self, other):
        return AddOP()(other, self)
    def __add__(self, other):
        return AddOP()(self, other)
    def __rsub__(self, other):
        return SubOP()(other, self)
    def __sub__(self, other):
        return SubOP()(self, other)
    def __rmul__(self, other):
        return MulOP()(other, self)
    def __mul__(self, other):
        return MulOP()(self, other)
    def __rtruediv__(self, other):
        return DivOP()(other, self)
    def __truediv__(self, other):
        return DivOP()(self, other)
    def __neg__(self):
        return MulOP()(self, -1)
    def __matmul__(self, other):
        return MatMulOP()(self, other)
    def __repr__(self):  # print触发
        # 如果self.op不是None, 说明张量是op得到的
        if self.op is not None:
            return f"tensor({self.data}, grad_fn=<{self.op.name}>)"
        else:
            return f"{self.data}"
    # 张量反向传播
    def backward(self, grad=1):
        # 梯度累加: 对于一个张量, 若多次运算, 梯度累加
        # grad是Superior transmission的张量 c梯度
        self.grad=(self.grad if self.grad else 0)+grad
        # 每个张量 c保存OP算子, 根据OP算子得到张量 c如何计算
        # 比如: c中AddOP，AddOP输入input(即a, b), 得到c=a+b
        # 已知grad_c, 递归计算grad_a和grad_b
        if self.op is not None:
            self.op.backward_native(grad)  # 用op, 递归计算     
# SessionRun 静态图中输入值, 计算OPS输出张量 c结果
# feed_dict输入字典传递, 如: {a: 1, b: 2}
def SessionRun(var, feed_dict):
    for key in feed_dict:
        key.data=feed_dict[key]
    return var.op.compute()
# 模拟包
class morch:
    @staticmethod
    def exp(value):
        return ExpOP()(value)
    @staticmethod
    def log(value):
        return LogOP()(value)
    @staticmethod
    def sum(value):
        return SumOP()(value)
    @staticmethod
    def mean(value):
        return MeanOP()(value)
5.3　机器学习自动微分
5.3.1　微分方法
# 与支持向量机一样，使用数值梯度检查作为调试工具
# 数值梯度应接近解析梯度
from cs231n.gradient_check import grad_check_sparse
f=lambda w: softmax_loss_naive(w, X_dev, y_dev, 0.0)[0]
grad_numerical=grad_check_sparse(f, W, grad, 10)
grad_check_sparse实现：
def grad_check_sparse(f, x, analytic_grad, num_checks=10, h=1e-5):
    // 充分利用一些随机元素，只返回此维度中的数值
    for i in range(num_checks):
        ix=tuple([randrange(m) for m in x.shape])

        oldval=x[ix]
        x[ix]=oldval+h # increment by h
        fxph=f(x) # evaluate f(x+h)
        x[ix]=oldval - h # increment by h
        fxmh=f(x) # evaluate f(x - h)
        x[ix]=oldval # reset
        grad_numerical=(fxph - fxmh)/(2 * h)
        grad_analytic=analytic_grad[ix]
        rel_error=(abs(grad_numerical - grad_analytic) /
                    (abs(grad_numerical)+abs(grad_analytic)))
        print('numerical: %f analytic: %f, relative error: %e'
              %(grad_numerical, grad_analytic, rel_error))
5.3.6　自动微分实现示例
实现代码如下所示：
class Node(object):
    // 计算图中的节点
    def __init__(self):
        // 构造函数中，新节点由算子对象__call__方法间接创建
            实例变量
            self.inputs: 输入节点的列表
            self.op: 关联的op对象，例如，如果此节点是通过添加其他两个节点创建的，则添加_op object。 
            self.const_attr: 加法或乘法常数
                例如，如果此节点由x+5创建，则self.const_attr=5。
            self.name: 用于调试的节点名称
        self.inputs=[]
        self.op=None
        self.const_attr=None
        self.name=""
    def __add__(self, other):
        // 添加两个节点，返回一个新节点
        if isinstance(other, Node):
            new_node=add_op(self, other)
        else:
            # 通过将常数存储在新节点的const_attr字段中
            # “other”参数是常数
            new_node=add_byconst_op(self, other)
        return new_node
    def __mul__(self, other):
        // 将两个节点相乘返回一个新节点
        if isinstance(other, Node):
            new_node=mul_op(self, other)
        else:
            new_node=mul_byconst_op(self, other)
        return new_node
    # 允许left-hand-side进行加法和乘法
    __radd__=__add__
    __rmul__=__mul__
    def __str__(self):
    // 允许打印以显示节点名称
        return self.name
    __repr__=__str__
// 自动求导代码如下所示：
def gradients(output_node, node_list):
    // 得到输出节点相对于node_list中每个节点的梯度
    Parameters
    output_node: 输出求导的节点
    node_list: 对wrt求导的节点列表
    Returns
    //梯度值列表，node_list中每个节点各一个
   
    # 从节点到每个输出节点的梯度贡献列表的映射
    # 辅助map存放node未布局的grad
    node_to_output_grads_list={}
    # 关于将output_node的梯度初始化为oneslike_op(output_node)的特别说明
    # 实际上是对标量reduce_sum(output_node)求导数，而不是向量output_node求导数。但这是损失函数的常见情况。
    node_to_output_grads_list[output_node]=[oneslike_op(output_node)]
    # 从节点到该节点梯度的映射
    # arranged后node的grad map梯度映射
    node_to_output_grad={}
    # 给定采用梯度wrt的output_node，以便进行逆拓扑顺序遍历图
    # 用先序遍历,获取reverse逆序元素顺序
    reverse_topo_order=reversed(find_topo_sort([output_node]))
    for node in reverse_topo_order:
        # 各种导数可能有多个, 将计算的导数相加
        grad=sum_node_list(node_to_output_grads_list[node])
        # 将计算的导数存入 node_to_output_grad的map中
        node_to_output_grad[node]=grad
        # 根据算子传递grad与当前node得到inputs的grads
        input_grads=node.op.gradient(node, grad)
        for id in range(len(node.inputs)):
            if node.inputs[id] not in node_to_output_grads_list:
                node_to_output_grads_list[node.inputs[id]]=[input_grads[id]]
            else:
              node_to_output_grads_list[node.inputs[id]].append(input_grads[id])
    grad_node_list=[node_to_output_grad[node] for node in node_list]
    return grad_node_list
x1=ad.Variable(name="x1")
x2=ad.Variable(name="x2")
y=x1 * x2+x1
grad_x1, grad_x2=ad.gradients(y, [x1, x2])
求损失权重的梯度时，将损失的向量转化为数字，对权重求梯度。
5.4　稀疏矩阵分析
下面是实现代码：
# 稠密到稀疏
from numpy import array
from scipy.sparse import csr_matrix
# 构建稠密矩阵
A=array([[1, 0, 0, 1, 0, 0], [0, 0, 2, 0, 0, 1], [0, 0, 0, 2, 0, 0]])
print(A)
# 转换为稀疏矩阵 (CSR method)
S=csr_matrix(A)
print(S)
# 重构稠密矩阵
B=S.todense()
print(B)
# 打印稠密数组，用CSR表示重新构建的稠密矩阵
[[1 0 0 1 0 0]
 [0 0 2 0 0 1]
 [0 0 0 2 0 0]]
 
 (0, 0) 1
 (0, 3) 1
 (1, 2) 2
 (1, 5) 1
 (2, 3) 2
 
[[1 0 0 1 0 0]
 [0 0 2 0 0 1]
 [0 0 0 2 0 0]]
NumPy计算矩阵的稀疏性。
计算矩阵的致密度与NumPy的非零元素，由count_nonzero给出元素的总数，由A.size给出数组大小。数组稀疏计算： 
sparsity=1.0 - count_nonzero(A)/A.size
如何计算数组的稀疏性，下面代码包括构建稠密矩阵，计算稀疏度，打印稀疏矩阵几个模块：
# 调用库函数
from numpy import array
from numpy import count_nonzero
# 构建稠密矩阵
A=array([[1, 0, 0, 1, 0, 0], [0, 0, 2, 0, 0, 1], [0, 0, 0, 2, 0, 0]])
print(A)
# 计算稀疏度
sparsity=1.0 - count_nonzero(A)/A.size
print(sparsity)
# 打印稀疏矩阵。
[[1 0 0 1 0 0]
 [0 0 2 0 0 1]
 [0 0 0 2 0 0]]
 
第6章　Relay IR中间表示
6.1 TVM数据介绍
6.1.2　Relay IR原理简介
实现代码如下所示：
def @relay_add_one(%x : Tensor((10,), f32)) {
    call_destination_passing @te_add_one(%x,  out=%b) 
} 

def @te_add_one(%a: NDArray, %b: NDArray) {
    var %n
    %A = decl_buffer(shape=[%n], src=%a)
    %B = decl_buffer(shape=[%n], src=%b) 
    // 主体ir含量需要改进
    for %i = 0 to 10 [data_par] {
        %B[%i] = %A[%i] + 1.0 
    }
}
6.1.3构建计算图
Relay将执行多功能组合。以下代码所示为两个函数调用示例：
def @muladd(%x, %y, %z) {
  %1=mul(%x, %y)
  %2=add(%1, %z)
  %2
}
def @myfunc(%x) {
  %1=@muladd(%x, 1, 2)
  %2=@muladd(%1, 2, 3)
  %2
}
def @myfunc(%x) {
  %1=equal(%x, 1)
   if (%1) {
      %x
   } else {
     %2=sub(%x, 1)
     %3=@myfunc(%2)
      %4=add(%3, %3)
      %4
  }
}
6.1.4　 Let绑定与作用域
6.3.1　添加节点，定义编译参数
在include/tvm/relay/attrs/中配置编译参数，添加节点。
在Python中，通过累积创建算子API接口，代码如下所示：
def cumprod(data, axis=None, dtype=None, exclusive=None):
// Numpy返回沿给定轴的元素的累积总和类型累积操作，返回元素沿给定轴包含乘积的累积
Parameters
data : relay.Expr
// 输入数据到算子
axis : int, optional
// 计算累加乘积的轴。默认值是在平铺阵列上计算累加积。
dtype : string, optional
// 返回数组和元素相乘的累加器的类型。
// 如果未指定数据类型，则用默认的数据类型。
exclusive : bool, optional
// 如果为true，返回不包含第一个元素的独占产品。换句话说，如果为true，则// 第j个输出元素将为第一个（j-1）元素的乘积。否则，将是第一个j元素。
// 零元素的乘积为1。
Returns
result : relay.Expr
// 如果axis为None，结果为一维数组结果的大小与数据相同，如果坐标轴不是None，形状与数据相同。
// 如果axis为None，结果为一维数组。
    
添加累加和对应累加积的API接口。可在include/tvm/relay/attrs/transform.h中配置性能时，再配置算子的位置，数据信息及特性，并当作结构体的合理字段。实现代码如下所示：
/*! \ cumsum和cumprod运算符中使用的简要属性 */
struct ScanopAttrs : public tvm::AttrsNode<ScanopAttrs> {
  Integer axis;
  DataType dtype;
  Bool exclusive=Bool(false);
  TVM_DECLARE_ATTRS(ScanopAttrs, "relay.attrs.ScanopAttrs") {
    TVM_ATTR_FIELD(axis).describe("The axis to operate over").set_default(NullValue<Integer>());
    TVM_ATTR_FIELD(dtype).describe("Output data type").set_default(NullValue<DataType>());
    TVM_ATTR_FIELD(exclusive)
        .describe("The first element is not included")
        .set_default(Bool(false));
  }
};
6.3.2　运算类型关系分析
根据算子实现运算类型规则，可分析乘法累积与加法求和运算符的类型关系，并在src/relay/op/tensor/transform.cc中得到，参考以下代码：
TVM_REGISTER_NODE_TYPE(ScanopAttrs);
bool ScanopRel(const Array<Type>& types, int num_inputs, const Attrs& attrs, const TypeReporter& reporter) {
    // types: [data, output]
    ICHECK_EQ(types.size(), 2) << "Expects two types, one for the input and another for the output";
    const auto* data=types[0].as<TensorTypeNode>();
    if (data == nullptr) {
        ICHECK(types[0].as<IncompleteTypeNode>())
        << "Scanop: expect input type to be TensorType but get " << types[0];
        return false;
    }
 
    const auto* param=attrs.as<ScanopAttrs>();
 
    auto dtype=param->dtype;
    if (dtype.is_void()) {
        dtype=data->dtype;
    }
 
    if (param->axis.defined()) {
        reporter->Assign(types[1], TensorType(data->shape, dtype));
    } else {
        auto prod=data->shape[0];
        for (size_t i=1; i < data->shape.size(); ++i) {
            prod=prod * data->shape[i];
        }
        reporter->Assign(types[1], TensorType({prod}, dtype));
    }
 
    return true;
}
6.3.3　C++中进行RELAY_REGISTER_OP宏注册
将下面代码加入到src/relay/op/tensor/transform中:
RELAY_REGISTER_OP("cumsum")
    .describe(
        R"doc(返回沿给定轴的元素累积和)doc" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(3)
    .add_type_rel("Cumsum", ScanopRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);
 
RELAY_REGISTER_OP("cumprod").describe(
        R"doc(返回元素沿给定轴的累积乘积)doc" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(3)
    .add_type_rel("Cumprod", ScanopRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);
6.3.4　算子注册与schedule调度
TVM定义算子运算和调度的组合策略，并对二维卷积进行深度计算。可在python/tvm/relay/op/strategy/generic.py与python/tvm/relay/op/strategy/cuda.py中，添加方法如下所示：
def wrap_compute_scanop(topi_compute):
    // 扫描topi计算封装wrapper扫描op样式topi计算
 
    def _compute_scanop(attrs, inputs, _):
        return [topi_compute(inputs[0], attrs.axis, attrs.dtype, attrs.exclusive)]
 
    return _compute_scanop
 
@override_native_generic_func("cumsum_strategy")
def cumsum_strategy(attrs, inputs, out_type, target):
    // cumsum通用策略
    strategy=_op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_scanop(topi.cumsum), 
        wrap_topi_schedule(topi.generic.schedule_extern), 
        name="cumsum.generic", 
  )
    return strategy
@override_native_generic_func("cumprod_strategy")
def cumprod_strategy(attrs, inputs, out_type, target):
    // cumprod通用策略
    strategy=_op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_scanop(topi.cumprod), 
        wrap_topi_schedule(topi.generic.schedule_extern), 
        name="cumprod.generic", 
  )
    return strategy
 
@cumsum_strategy.register(["cuda", "gpu"])
def cumsum_strategy_cuda(attrs, inputs, out_type, target):
    // cumsum cuda策略
    strategy=_op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_scanop(topi.cuda.cumsum), 
        wrap_topi_schedule(topi.cuda.schedule_scan), 
        name="cumsum.cuda", 
  )
    return strategy
  
@cumprod_strategy.register(["cuda", "gpu"])
def cumprod_strategy_cuda(attrs, inputs, out_type, target):
    // cumprod cuda策略
    strategy=_op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_scanop(topi.cuda.cumprod), 
        wrap_topi_schedule(topi.cuda.schedule_scan), 
        name="cumprod.cuda", 
  )
    return strategy
具体代码如下所示：
# cumsum
@_reg.register_compute("cumsum")
def compute_cumsum(attrs, inputs, output_type):
    // cumsum计算定义
    return [topi.cumsum(inputs[0], attrs.axis, attrs.dtype, attrs.exclusive)]
 
_reg.register_strategy("cumsum", strategy.cumsum_strategy)
_reg.register_shape_func("cumsum", False, elemwise_shape_func)
 
# cumprod
@_reg.register_compute("cumprod")
def compute_cumprod(attrs, inputs, output_type):
    // 计算定义
return [topi.cumprod(inputs[0], attrs.axis, attrs.dtype, attrs.exclusive)]
 
_reg.register_strategy("cumprod", strategy.cumprod_strategy)
_reg.register_shape_func("cumprod", False, elemwise_shape_func)
 
6.3.5　注册函数API分析
使用Op::Get从算子注册中获取算子信息，并将参数传给调用节点。
而实现代码在src/relay/op/tensor/transform.cc中，下面是MakeCumsum，MakeCumprod，TVM_REGISTER_GLOBAL的代码：
Expr MakeCumsum(Expr data, Integer axis, DataType dtype, Bool exclusive) {
    auto attrs=make_object<ScanopAttrs>();
    attrs->dtype=dtype;
    attrs->axis=axis;
    attrs->exclusive=exclusive;
    static const Op& op=Op::Get("cumsum");
    return Call(op, {data}, Attrs(attrs), {});
}
 
TVM_REGISTER_GLOBAL("relay.op._make.cumsum").set_body_typed(MakeCumsum);
Expr MakeCumprod(Expr data, Integer axis, DataType dtype, Bool exclusive) {
    auto attrs=make_object<ScanopAttrs>();
    attrs->dtype=dtype;
    attrs->axis=axis;
    attrs->exclusive=exclusive;
    static const Op& op=Op::Get("cumprod");
    return Call(op, {data}, Attrs(attrs), {});
}
 
TVM_REGISTER_GLOBAL("relay.op._make.cumsum").set_body_typed(MakeCumprod);
通过使用relay.op._make.cumsum(...) 与 relay.op._make.cumsumprod(...)，TVM_REGISTER_GLOBAL 开放MakeCumsum 与MakeCumprod函数API接口。
6.3.6　将Python API打包
通过TVM_REGISTER_GLOBAL导出函数，再封装在Python函数中，但不直接在Python中调用。而在python/tvm/relay/op/transform.py中，却开放了API接口。代码如下所示：
def cumsum(data, axis=None, dtype=None, exclusive=None):
    return _make.cumsum(data, axis, dtype, exclusive)
def cumprod(data, axis=None, dtype=None, exclusive=None):
    return _make.cumprod(data, axis, dtype, exclusive)
Python包向算子提供简单API接口。例如，实现concat算子注册，并将张量元组与Python包装器组合成元组。示例代码如下所示：
def concat(*args):
    // 沿轴连接输入张量
    Parameters
    args: //张量列表
 
    Returns
    tensor: //串联张量
    
    tup=Tuple(list(args))
    return _make.concat(tup)
6.3.7　单元测试分析
单元tests可在tests/python/relay/test_op_level3.py中查到，并用于累加和与积算子。
1. 梯度算子
在Python中添加Python梯度算子，并可在Python/tvm/relay/op/_tensor_grad.py中，下面是sigmoid乘法代码示例：
@register_gradient("sigmoid")
def sigmoid_grad(orig，grad):
    // Returns [grad * sigmoid(x) * (1 - sigmoid(x))].
    return [grad * orig * (ones_like(orig) - orig)]
具体代码如下所示：
@register_gradient("multiply")
def multiply_grad(orig, grad):
    // Returns [grad * y, grad * x]
    x, y=orig.args
    return [collapse_sum_like(grad * y, x), 
            collapse_sum_like(grad * x, y)]
	 
2. 在C++中增加梯度
代码如下所示：
tvm::Array<Expr> MultiplyGrad(const Expr& orig_call, const Expr& output_grad) {
    const Call& call=orig_call.Downcast<Call>();
    return { CollapseSumLike(Multiply(output_grad, call.args[1]), call.args[0]), CollapseSumLike(Multiply(output_grad, call.args[0]), call.args[1]) 
};
}
注册代码如下所示：
RELAY_REGISTER_OP("multiply")
    // ...
    // 设置其他属性
    // ...
    .set_attr<FPrimalGradient>("FPrimalGradient", MultiplyGrad);
 6.4　TVM中IR示例
6.4.1　 IRModule技术分析
下面是实现代码：
Global_var
import tvm
from tvm import relay
import numpy as np
# step 1: 建模
m,n = 4, 2
x = relay.var("x", shape=(m,n), dtype='float32')
out = relay.nn.softmax(x)
net = relay.Function([x], out)

# step 2: 编译降级（Lowering）下译（Lowering）
module = tvm.IRModule.from_expr(net)
lib = relay.build(module, "llvm")

# step 3: 输入张量数据
ctx = tvm.cpu(0)
x_t = tvm.nd.array(np.random.uniform(size=[m,n]).astype('float32'), ctx)
runtime = tvm.contrib.graph_runtime.GraphModule(lib["default"](ctx))
runtime.set_input("x", x_t)
runtime.run()
print(runtime.get_output(0))

# print(net.body)

fn (%x: Tensor[(4, 2), float32]) {
  nn.softmax(%x)
}

# print(module)

def @main(%x: Tensor[(4, 2), float32]) {
  nn.softmax(%x)
}
6.4.2　TVM Runtime运行时分析
实现代码如下所示：
import tvm
# Python运行时执行程序示例，包括类型注释
mod: tvm.runtime.Module = tvm.runtime.load_module("compiled_artifact.so")
arr: tvm.runtime.NDArray = tvm.nd.array([1, 2, 3], ctx=tvm.gpu(0))
fun: tvm.runtime.PackedFunc = mod["addone"]
fun(a)
print(a.asnumpy())
运行时的3大核心模块包括：
1）	runtime.Module：封装编译DSO（dynamic share object，动态共享对象），包含PackedFunc，并用名称获取函数并用name获取function。
2）	runtime.PackedFunc：后端生成深度学习的KernelFunc函数。
3）	runtime.NDArray：封装张量。
6.4.3　预测部署实现
下面是实现代码：
import tvm
import numpy as np

n = 12
A = te.placeholder((n,), name="A") # 张量
B = te.compute(A.shape, lambda *i: A(*i) + 1.0, name="B") # 张量
C = te.compute(A.shape, lambda *i: A(*i) - 1.0, name="C") # 张量
s = te.create_scheduleC[B.op, C.op])  # schedule
add_func = tvm.build(s, [A, B, C], "llvm", name="add") # compile
# 准备数据
ctx = tvm.cpu(0)
a_t = tvm.nd.array(np.random.uniform(size=nn).astype(A.type), ctx)
b_t = tvm.nd.array(np.zeros(nn, dtype=A.dtype), ctx)
c_t = tvm.nd.array(np.zeros(nn, dtype=A.dtype), ctx)
add_func(a_t, b_t, c_t)
# 对于预测部署，可将计算逻辑编译为DSO
from tvm.contrib import cc
# 序列化
add_func.save('./add_kernel.o')
cc.create_shared('./for_infer.so', ['./add_kernel.o'])
# load for inference
m = tvm.runtime.load_module('./for_infer.so')
add_func = m['add']  # load add kernel func
add_func(a_t, b_t, c_t)  # infer
# model序列化与加载示例：
# Resnet18 workload
resnet18_mod, resnet18_params = relay.testing.resnet.get_workload(num_layers=18)
# build
with relay.build_config(opt_level=3):
    _, resnet18_lib, _ = relay.build_module.build(resnet18_mod, "cuda", params=resnet18_params)
# export library
file_name = "./deploy.so"
resnet18_lib.export_library(file_name)
# 重新加载
loaded_lib = tvm.runtime.load_module(file_name)
# inference
data = np.random.uniform(-1, 1, size=input_shape(mod)).astype("float32")
ctx = tvm.gpu()
gmod = graph_runtime.GraphModule(loaded_lib["default"](ctx))
gmod.set_input("data", data)
gmod.run()
out = gmod.get_output(0).asnumpy()
6.4.4　动态图实shape实现
代码如下所示：
import tvm
import numpy as np
# 组网
n, m = te.size_var("n"), te.size_var("m")
A = te.placeholder((n,m), name="A")
k = te.reduce_axis((0, m), "k")
B = te.compute((n,),lambda i:te.sum(A[i,k], axis=k), name="B")
# 编译
s = te.create_schedule(B.op)
net = tvm.build(s, [A, B, n, m])
# 执行
def run(n, m):
  ctx = tvm.cpu(0)
  a = tvm.nd.array(np.random.uniform(size=[n,m]).astype(A.dtype), ctx)
  b = tvm.nd.array(np.zeros((n,)).astype(A.dtype), ctx)
  return net(a, b, n, m)

run(4, 6)
run(10, 16)
# TVM包括debug机制，可打印中间编译代码：
print(str(tvm.lower(s, [A, B])))

primfn(A_1: handle, B_1: handle) -> ()
  attr = {"global_symbol": "main", "tir.noalias": True}
  buffers = {B: Buffer(B_2: Pointer(float32), float32, [n: int32], [stride: int32], type="auto"),
             A: Buffer(A_2: Pointer(float32), float32, [n, m: int32], [stride_1: int32, stride_2: int32], type="auto")}
  buffer_map = {A_1: A, B_1: B} {
  for (i: int32, 0, n) {
    B_2[(i*stride)] = 0f32
    for (k: int32, 0, m) {
      B_2[(i*stride)] = ((float32*)B_2[(i*stride)] + (float32*)A_2[((i*stride_1) + (k*stride_2))])
    }
  }
}
可用print(m.get_source())，查看构建后build后的LLVM程序。
6.5.2　CUDA编程模型基础理论
相应代码如下所示：
// 内核定义
__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N]) 
{ 
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y; 
    if (i < N && j < N) 
        C[i][j] = A[i][j] + B[i][j]; 
}
int main() 
{ 
    ...
    // 内核线程配置
    dim3 threadsPerBlock(16, 16); 
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    // kernel调用
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C); 
    ...
}
CUDA的memory模型，如图6.15所示。而Thread有多种内存，包括本地内存、共享内存、全局内存、常量内存与结构内存。由于内核的thread结构层次复杂，因此内核可执行很多thread调度。而GPU核心模块SM，全称流Multiprocessor，即流式多处理器。SM可调度多个线程块，而内核可分配多个SM。这里grid是逻辑层，SM是物理层。SM采用SIMT架构，并执行unit是线程包，而且线程包共包含32个thread。
在CUDA编码前，先查找GPU的硬件配置，可得到GPU 配置属性如下所示：
int dev = 0;
    cudaDeviceProp devProp;
    CHECK(cudaGetDeviceProperties(&devProp, dev));
    std::cout << "使用GPU device " << dev << ": " << devProp.name << std::endl;
    std::cout << "SM的数量: " << devProp.multiProcessorCount << std::endl;
    std::cout << "各thread block的共享memory大小: " << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "各thread block最大thread number: " << devProp.maxThreadsPerBlock << std::endl;
    std::cout << "各EM最大thread number: " << devProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "各SM的最大thread束数: " << devProp.maxThreadsPerMultiProcessor /32<< std::endl;
6.5.3　实现向量加法实例
先定义内核：
// 两个向量加法的内核, grid与block均为1-dim
__global__ void add(float* x, float * y, float* z, int n)
{
    //获取全局索引
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    // 步长
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
    {
        z[i] = x[i] + y[i];
    }
}
实现向量加法：
int main()
{
    int N = 1 << 20;
    int nBytes = N * sizeof(float);
    // 申请host内存
    float *x, *y, *z;
    x = (float*)malloc(nBytes);
    y = (float*)malloc(nBytes);
    z = (float*)malloc(nBytes);

    // 初始化数据
    for (int i = 0; i < N; ++i)
    {
        x[i] = 10.0;
        y[i] = 20.0;
    }

    // 申请设备内存
    float *d_x, *d_y, *d_z;
    cudaMalloc((void**)&d_x, nBytes);
    cudaMalloc((void**)&d_y, nBytes);
    cudaMalloc((void**)&d_z, nBytes);

    // 将host数据拷贝到设备
    cudaMemcpy((void*)d_x, (void*)x, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_y, (void*)y, nBytes, cudaMemcpyHostToDevice);
    // 定义内核的执行配置
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    // 执行内核
    add << < gridSize, blockSize >> >(d_x, d_y, d_z, N);

    // 将device结果copy到host 
    cudaMemcpy((void*)z, (void*)d_z, nBytes, cudaMemcpyDeviceToHost);

    // 检查执行结果
    float maxError = 0.0;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(z[i] - 30.0));
    std::cout << "最大误差: " << maxError << std::endl;

    // 释放device内存
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    // 释放host内存
    free(x);
    free(y);
    free(z);

    return 0;
}
这是CUDA用cudaMallocManaged配置托管内存：
cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flag=0);
利用统一host与device内存，具体程序简化如下：
int main()
{
    int N = 1 << 20;
    int nBytes = N * sizeof(float);
    // 申请托管内存
    float *x, *y, *z;
    cudaMallocManaged((void**)&x, nBytes);
    cudaMallocManaged((void**)&y, nBytes);
    cudaMallocManaged((void**)&z, nBytes);
    // 初始化数据
    for (int i = 0; i < N; ++i)
    {
        x[i] = 10.0;
        y[i] = 20.0;
    }
    // 定义内核的执行配置
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    // 执行内核
    add << < gridSize, blockSize >> >(x, y, z, N);
    // 同步device 确保results能正确访问
    cudaDeviceSynchronize();
    // 检查执行结果
    float maxError = 0.0;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(z[i] - 30.0));
    std::cout << "最大误差: " << maxError << std::endl;
    // 释放内存
    cudaFree(x);
    cudaFree(y);
    cudaFree(z);
    return 0;
}
6.5.4　实现矩阵乘法实例
设置matrix结构体如下所示：
// 矩阵类型, 行优先, M(row, col) = *(M.elements + row * M.width + col)
struct Matrix
{
    int width;
    int height;
    float *elements;
};
这里实现矩阵相乘核函数，同时定义辅助__device__函数，以便获取matrix元素与元素赋值。代码如下所示：
// 取得矩阵A的(row, col)元素
__device__ float getElement(Matrix *A, int row, int col)
{
	return A->elements[row * A->width + col];
}

// 矩阵A(row, col)的elements赋值
__device__ void setElement(Matrix *A, int row, int col, float value)
{
	A->elements[row * A->width + col] = value;
}

// matrix乘法内核, 2-D, 各thread算出element
__global__ void matMulKernel(Matrix *A, Matrix *B, Matrix *C)
{
	float Cvalue = 0.0;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	for (int i = 0; i < A->width; ++i)
	{
		Cvalue += getElement(A, row, i) * getElement(B, i, col);
	}
	setElement(C, row, col, Cvalue);
}
用统一内存开发矩阵乘法示例：
int main()
{
    int width = 1 << 10;
    int height = 1 << 10;
    Matrix *A, *B, *C;
    // 申请托管内存
    cudaMallocManaged((void**)&A, sizeof(Matrix));
    cudaMallocManaged((void**)&B, sizeof(Matrix));
    cudaMallocManaged((void**)&C, sizeof(Matrix));
    int nBytes = width * height * sizeof(float);
    cudaMallocManaged((void**)&A->elements, nBytes);
    cudaMallocManaged((void**)&B->elements, nBytes);
    cudaMallocManaged((void**)&C->elements, nBytes);
    // 初始化数据
    A->height = height;
    A->width = width;
    B->height = height;
    B->width = width;
    C->height = height;
    C->width = width;
    for (int i = 0; i < width * height; ++i)
    {
        A->elements[i] = 1.0;
        B->elements[i] = 2.0;
    }
    // 定义内核的执行配置
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
        (height + blockSize.y - 1) / blockSize.y);
    // 执行内核
    matMulKernel << < gridSize, blockSize >> >(A, B, C);

    // 同步device 确保results正确访问
    cudaDeviceSynchronize();
    // 检查执行结果
    float maxError = 0.0;
    for (int i = 0; i < width * height; ++i)
        maxError = fmax(maxError, fabs(C->elements[i] - 2 * width));
    std::cout << "最大误差: " << maxError << std::endl;
    return 0;
}
 
第7章　Code Generation代码生成
7.2.2　进行codegenCodeGen分析
在C中可手动实现两个宏如下所示：
#define CSOURCE_BINARY_OP_1D(p_ID_, p_OP_, p_DIM1_)         
    extern "C" void p_ID_(float* a, float* b, float* out) { 
        for (int64_t i=0; i < p_DIM1_; ++i) {             
            out[i]=a[i] p_OP_ b[i];                       
        }                                                   
    }

#define CSOURCE_BINARY_OP_2D(p_ID_, p_OP_, p_DIM1_, p_DIM2_)  
    extern "C" void p_ID_(float* a, float* b, float* out) {   
        for (int64_t i=0; i < p_DIM1_; ++i) {               
            for (int64_t j=0; j < p_DIM2_; ++j) {           
                int64_t k=i * p_DIM2_+j;                  
                out[k]=a[k] p_OP_ b[k];                     
            }                                                 
        }                                                     
    }
　　
两个宏生成二元算子。实现代码如下所示：
c_compiler_input0
       |
      add <-- c_compiler_input1
       |
    subtract <-- c_compiler_input2
       |
    multiply <-- c_compiler_input3
       |
      out
// 生成编译代码，执行子图：
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/packed_func.h>
#include <dlpack/dlpack.h>
#include <cstdint>
#include <cstring>
#include <iostream>

#define GCC_BINARY_OP_1D(p_ID_, p_OP_, p_DIM1_)           
  extern "C" void p_ID_(float* a, float* b, float* out) { 
    for (int64_t i=0; i < p_DIM1_; ++i) {               
      out[i]=a[i] p_OP_ b[i];                           
    }                                                     
  }

#define GCC_BINARY_OP_2D(p_ID_, p_OP_, p_DIM1_, p_DIM2_)  
  extern "C" void p_ID_(float* a, float* b, float* out) { 
    for (int64_t i=0; i < p_DIM1_; ++i) {               
      for (int64_t j=0; j < p_DIM2_; ++j) {             
        int64_t k=i * p_DIM2_+j;                      
        out[k]=a[k] p_OP_ b[k];                         
      }                                                   
    }                                                     
  }

// 注1
GCC_BINARY_OP_2D(gcc_0_0, *, 10, 10);
GCC_BINARY_OP_2D(gcc_0_1, -, 10, 10);
GCC_BINARY_OP_2D(gcc_0_2, +, 10, 10);
// 注2
extern "C" void gcc_0_(float* gcc_input0, float* gcc_input1, 
                       float* gcc_input2, float* gcc_input3, float* out) {
  float* buf_0=(float*)malloc(4 * 100);
  float* buf_1=(float*)malloc(4 * 100);
  gcc_0_2(gcc_input0, gcc_input1, buf_0);
  gcc_0_1(buf_0, gcc_input2, buf_1);
  gcc_0_0(buf_1, gcc_input3, out);
  free(buf_0);
  free(buf_1);
}

// 注3
extern "C" int gcc_0_wrapper(DLTensor* arg0, DLTensor* arg1, DLTensor* arg2, 
                             DLTensor* arg3, DLTensor* out) {
  gcc_0_(static_cast<float*>(arg0->data), static_cast<float*>(arg1->data), 
         static_cast<float*>(arg2->data), static_cast<float*>(arg3->data), 
         static_cast<float*>(out->data));
  return 0;
}
TVM_DLL_EXPORT_TYPED_FUNC(gcc_0, gcc_0_wrapper);
7.2.3　实现 Codegen_CCodeGenC举例
在src/relay/backend/contrib/codegen_cCodeGenC/codegenCodeGen.cc中，首先在tvm.relay.contrib命名空间下创建一个codegenCodeGen类框架代码如下所示：
tvm.relay.contrib：
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/object.h>
#include <fstream>
#include <sstream>
#include "codegen_cCodeGenC.h"

namespace tvm {
namespace relay {
namespace contrib {
class CodegenCodeGenC : public ExprVisitor，public CodegenCodeGenCBase {
  public:
    explicit CodegenCodeGenC(const std::string& id) { this->ext_func_id_=id; }
    void VisitExpr_(const VarNode* node) { ; }
    void VisitExpr_(const CallNode* call) final { ; }
    std::string JIT() { ; }
  private:
    /*! \简述表示C源函数的函数id */
    std::string ext_func_id_="";
    /*! \简要介绍包装C函数的索 */
    int func_idx=0;
    /*! \简要介绍已分配缓冲区的索引 */
    int buf_idx_=0;
    /*! \简要介绍与C编译器兼容的函数的参数 */
    std::vector<std::string> ext_func_args_;
    /*! \简要介绍C编译器兼容函数的语句 */
    std::vector<std::string> ext_func_body;
    /*! \简要介绍C编译器兼容函数的声明语句 */
    std::vector<std::string> func_decl_;
    /*! \简述缓冲区的声明 */
    std::vector<std::string> buf_decl_;
    /*! \简要介绍输出的名称和索引对 */
    std::vector<std::pair<std::string, int>> out_;
}
7.3　算子的codegenCodeGen原理与示例
7.3.1　生成函数声明
幸运的是，可以从【CallNode】位置轻松获取此如下所示信息：
// 生成唯一函数名
std::string func_name=ext_func_id_+"_"+std::to_string(func_idx++);
// 生成函数声明字符串
macro_stream << "CSOURCE_BINARY_OP_" << call->args.size() << "D(" << func_name << ", ";
// 检查运算符类型
if (IsOp(call, "add")) {
  macro_stream << "+";
} else if (IsOp(call, "subtract")) {
  macro_stream << "-";
} else if (IsOp(call, "multiply")) {
  macro_stream << "*";
} else {
  LOG(FATAL) << "Unrecognized op";
}
// 提取输入张量shape
auto in_shape=GetShape(call->args[0]->checked_type());
for (size_t i=0; i < in_shape.size(); ++i) {
  macro_stream << ", " << in_shape[i];
}
macro_stream << ");";
func_decl_.push_back(macro_stream.str());
7.3.2　生成函数调用
实现代码如下所示：
bool first=true;
decl_stream << func_name << "(";
for (size_t i=0; i < call->args.size(); ++i) {
  VisitExpr(call->args[i]); // 注1
  for (auto out : out_) {
    if (!first) {
      decl_stream << ", ";
    }
    first=false;
    decl_stream << out.first;
  }
}
#define CSOURCE_BINARY_OP_1D(p_ID_, p_OP_, p_DIM1_)         
    extern "C" void p_ID_(float* a, float* b, float* out) { 
        for (int64_t i=0; i < p_DIM1_; ++i) {             
            out[i]=a[i] p_OP_ b[i];                       
        }                                                   
    }
#define CSOURCE_BINARY_OP_2D(p_ID_, p_OP_, p_DIM1_, p_DIM2_)  
    extern "C" void p_ID_(float* a, float* b, float* out) {   
        for (int64_t i=0; i < p_DIM1_; ++i) {               
            for (int64_t j=0; j < p_DIM2_; ++j) {           
                int64_t k=i * p_DIM2_+j;                  
                out[k]=a[k] p_OP_ b[k];                     
            }                                                 
        }                                          
  arg_node                 arg_node <- Visit arg (Note 1)       arg_node
     |                        |                                    |
 curr_node <- Process      curr_node                            curr_node <- Put "buf_0" as an input buffer

(a) out_={}               
(b) out_={}                          
(c) out_={("buf_0", 20)}
            
}
7.3.3　生成输出缓冲区
实现代码如下所示：
// 支持一个输出
auto type_node=call->checked_type().as<TensorTypeNode>();
ICHECK(type_node != nullptr && runtime::TypeMatch(type_node->dtype, kDLFloat, 32)) << "Only support single output tensor with float type";
// 生成一个buffer名称
std::string out="buf_"+std::to_string(buf_idx_++);
// 将形状提取为buffer大小
auto out_shape=GetShape(call->checked_type());
int out_size=1;
for (size_t i=0; i < out_shape.size(); ++i) {
  out_size *= out_shape[i];
}
// 进行缓冲区分配与推送
buf_stream << "float* " << out << "=(float*)std::malloc(4 * " << out_size << ");";
buf_decl_.push_back(buf_stream.str());
// 分配输出缓冲区，关闭函数调用字符串，推送生成的函数调用到类变量ext_func_body。
decl_stream << ", " << out << ");";
ext_func_body.push_back(decl_stream.str());
// 分配输出缓冲区后，可关闭函数调用字符串，将生成的函数调用推送到类变量ext_func_body。
decl_stream << ", " << out << ");";
ext_func_body.push_back(decl_stream.str());
7.3.4　更新输出缓冲区
为了让接受当前调用节点的输出，作为输入的下一个节点，知道应使用的缓冲区，需要在离开此访问函数前更新类变量out_：
out_.clear();
out_.push_back({out, out_size});
1. 输入变量的codegenCodeGen
VarNode表示模型中的输入张量。拥有的唯一的，但重要的信息是名称提示（如data，weight等）。在访问VarNode时，只需更新类变量out_，传递名称提示，以便后代调用节点，可以生成正确的函数调用。实现代码如下所示： 
void VisitExpr_(const VarNode* node) {
  ext_func_args_.push_back(node->name_hint());
  out_.clear();
  out_.push_back({node->name_hint(), 0});
}
2. 代码发布
可以调用JitImpl如下：
JitImpl("gcc_0" /* Subgraph symbol (ID) */, 
        {"gcc_input0", "gcc_input1", "gcc_input2", "gcc_input3"} /* Input arguments输入参数 */, 
        {"float *buf_0=(float*)malloc(4 * 20)", ...} /* Buffer allocations 缓存分配*/, 
        {"gcc_0_2(gcc_input0, gcc_input1, buf_0);"} /* Function body 函数体*/, 
        {"out"} /* Output */);
因此，JIT实现过程中唯一需要做的就是将生成的所有子图函数代码，传递给JitImpl，代码如下所示：
std::string JIT() {
  // 写宏
  for (auto decl : func_decl_) {
    code_stream_ << decl << "\n";
  }
  return JitImpl(ext_func_id_, ext_func_args_, buf_decl_, ext_func_body, out_);
}
传递的所有的变量（ext_func_id等）都是类变量，在遍历子图时会被填充。
3. 实现 CSourceCodegenCodeGen基类
    同样，创建一个类框架并实现所需的功能。继承CSourceModuleCodegenCodeGenBase。代码如下所示：
class CSourceCodegenCodeGen : public CSourceModuleCodegenCodeGenBase {
 public:
// 传递子图函数，生成C代码
  void GenCFunc(const Function& func) { ; }
// 使用GenCFunc生成C代码，打包成C模型
  runtime::Module CreateCSourceModule(const NodeRef& ref) override { ; }
 private:
  std::ostringstream code_stream_;
};
4. 用GenCFunc生成C代码
代码如下所示：
void GenCFunc(const Function& func) {
  ICHECK(func.defined()) << "Input error: expect a Relay function.";
// 记录运行时查找表的外部信号
  auto sid=GetExtSymbol(func);
  CodeGenCodeGenC builder(sid);
  builder.VisitExpr(func->body);
  code_stream_ << builder.JIT();
}
5. 构建CSourceModule，编译CodegenCodeGen
实现代码如下所示： 
1）创建CSourceModule
runtime::Module CreateCSourceModule(const NodeRef& ref) override {
  // 创建headers
  code_stream_ << "#include <cstdint>\n";
  code_stream_ << "#include <iostream>\n";
  code_stream_ << "#include <cstdlib>\n";
  code_stream_ << "#include <stdio.h>\n";
  code_stream_ << "#include <cstring>\n";
  code_stream_ << "#include <tvm/runtime/c_runtime_api.h>\n";
  code_stream_ << "#include <dlpack/dlpack.h>\n";
  // 为运算符定义附加一些常用宏
  const char* operator_macro=R"op_macro(
  #define CSOURCE_BINARY_OP_1D(p_ID_, p_OP_, p_DIM1_)       
    extern "C" void p_ID_(float* a, float* b, float* out) { 
      for (int64_t i=0; i < p_DIM1_; ++i) {               
        out[i]=a[i] p_OP_ b[i];                           
      }                                                     
    }
  #define CSOURCE_BINARY_OP_2D(p_ID_, p_OP_, p_DIM1_, p_DIM2_)  
    extern "C" void p_ID_(float* a, float* b, float* out) {     
      for (int64_t i=0; i < p_DIM1_; ++i) {                   
        for (int64_t j=0; j < p_DIM2_; ++j) {                 
          int64_t k=i * p_DIM2_+j;                          
          out[k]=a[k] p_OP_ b[k];                             
        }                                                       
      }                                                         
    }
)op_macro";
  code_stream_ << operator_macro << "\n\n";
  // 为子图生成C代码
  if (ref->IsInstance<FunctionNode>()) {
    GenCFunc(Downcast<Function>(ref));
  } else if (ref->IsInstance<relay::ModuleNode>()) {
    relay::Module mod=Downcast<relay::Module>(ref);
    for (const auto& it : mod->functions) {
      GenCFunc(Downcast<Function>(it.second));
    }
  } else {
    LOG(FATAL) << "The input ref is expected to be a Relay function or module"
               << "\n";
  }
  // 创建CSourceModule
  const auto* pf=runtime::Registry::Get("module.csource_module_create");
  ICHECK(pf != nullptr) << "Cannot find csource module to create the external runtime module";
  return (*pf)(code_stream_.str(), "cc");
}
2）注册 CodegenCodeGen
代码如下所示：
runtime::Module CCompiler(const NodeRef& ref) {
  CSourceCodegenCodeGen csource;
  return csource.CreateCSourceModule(ref);
}
// 将函数注册到 TVM 后端，代码如下所示：
TVM_REGISTER_GLOBAL("relay.ext.ccompiler").set_body_typed(CCompiler);
ccompiler是定制标记，进行子图ccompiler注释，标记生成和卸载子图。
CMcmake 配置include标志。创建cmake文件cmake/modules/contrib/CODEGENCODEGENC.cmake，代码如下所示：
if(USE_CODEGENCODEGENC)
  file(GLOB CSOURCE_RELAY_CONTRIB_SRC src/relay/backend/contrib/codegen_cCodeGenC/codegenCodeGen.cc)
  list(APPEND COMPILER_SRCS ${CSOURCE_RELAY_CONTRIB_SRC})
endif(USE_CODEGENCODEGENC)
使用config.cmake配置 TVM 时，包含以下编译器：set(USE_CODEGENCODEGENC ON)。
3）表示层实施 CodegenCodeGen
尽管已经演示了如何实现C代码生成，但是硬件可能需要其他的图形表示形式，如JSON。在这种情况下，可以修改CodegenCodeGenC类，已经实现了自主生成的图形表示，实现定制的运行时模块，使TVM运行时，如何执行该图形表示。
为了简化，定义了一个名为“ ExampleJSON”的图表示。ExampleJSON并不是真正的JSON，仅仅是没有控制流的图的简单表示。例如，假设有一个名为subgraph_0的子图如下所示：
 input0
   |
  add <-- input1
   |
subtract <-- input2
   |
multiply <-- input3
   |
  out
子图的 ExampleJSON如下所示：
subgraph_0
  input 0 10 10
  input 1 10 10
  input 2 10 10
  input 3 10 10
  add 4 inputs: 0 1 shape: 10 10
  sub 5 inputs: 4 2 shape: 10 10
  mul 6 inputs: 5 3 shape: 10 10
关键字声明输入张量的ID和形状; 其他语句以语法描述计算:
<op> <output ID> inputs: [input ID] shape: [shape]
目标是实现以下定制的TVM运行时模块，执行ExampleJSON图。
实现代码如下所示：
runtime::Module ExampleJsonCompiler(const NodeRef& ref) {
    ExampleJsonCodeGenCodeGen codegenCodeGen(ref);
std::string code=codegenCodeGen.gen(); 
//注1
const auto* pf=runtime::Registry::Get("module.examplejson_module_create");
 //注2
    ICHECK(pf != nullptr) << "Cannot find ExampleJson module to create the external runtime module";
    return (*pf)(code);
}
TVM_REGISTER_GLOBAL("relay.ext.examplejsoncompiler").set_body_typed(ExampleJsonCompiler);
CodeGen类实现如下所示：
1）获得ExampleJsonCodeGenCodeGen
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/object.h>
#include <fstream>
#include <sstream>

namespace tvm {
namespace relay {
namespace contrib {

class ExampleJsonCodeGenCodeGen : public ExprVisitor {
  public:
    explicit ExampleJsonCodeGenCodeGen();
    //注1
    void VisitExpr_(const VarNode* node) { /* 在此示例中跳过Skip in this example. */ }
    void VisitExpr_(const CallNode* call) final { /* 在此示例中跳过Skip in this example. */ }
    //注2
    std::string gen(NodeRef& ref) {
        this->code="";
        if (ref->IsInstance<FunctionNode>()) {
            this->visit(Downcast<Function>(ref));
        } else if (ref->IsInstance<relay::ModuleNode>()) {
            relay::Module mod=Downcast<relay::Module>(ref);
            for (const auto& it : mod->functions) {
                this->visit(Downcast<Function>(it.second));
            }
        } else {
            LOG(FATAL) << "The input ref is expected to be a Relay function or module";
        }
        return this->code;
    }
  private:
      /*! \简述表示C源函数的函数id */
     std::string code;
}
定义一个自定义的运行时类，如下所示。该类必须从TVM派生ModuleNode，以便与其他TVM运行时模块兼容。实现代码如下所示： 
#include <dmlc/logging.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <fstream>
#include <cmath>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace tvm {
namespace runtime {
class ExampleJsonModule : public ModuleNode {
 public:
  explicit ExampleJsonModule(std::string graph_json);
  PackedFunc GetFunction(const std::string& name, 
                         const ObjectPtr<Object>& sptr_to_self) final;
  const char* type_key() const { return "examplejson"; }
  void SaveToBinary(dmlc::Stream* stream) final;
  static Module LoadFromBinary(void* strm);
  static Module Create(const std::string& path);
  std::string GetSource(const std::string& format="");
  void Run(int id, const std::vector<int>& inputs, int output);
  void ParseJson(const std::string& json);

 private:
  /* \简述表示计算图的json字符串 */
  std::string graph_json_;
  /* \简述正在处理的子图 */
  std::string curr_subgraph_;
  /* ! \简要介绍从子图id到节点条目的简单图 */
  std::map<std::string, std::vector<NodeEntry> > graph_;
  /* \简要介绍一个简单的池化，包括图中每个节点的张量 */
  std::vector<NDArray> data_entry_;
  /* \简述从节点id到操作名称的映射 */
  std::vector<std::string> op_id_;
};

7. 实现构造函数示例分析
构造函数示例实现代码如下所示：
explicit ExampleJsonModule(std::string graph_json) {
  this->graph_json_=graph_json;
  ParseJson(this->graph_json_);
}
然后，实现ParseJson解析ExampleJSON格式的子图，在内存中构造一个图供以后使用。由于在此示例中不支持带有分支的子图，因此仅使用数组按顺序存储子图中的每个节点。实现代码如下所示：
void ParseJson(const std::string& json) {
  std::string line;
  std::string curr_subgraph;
  std::stringstream ss(json);

  while (std::getline(ss, line, '\n')) {
    std::stringstream ss2(line);
    std::string token;
    int id=0;
    ss2 >> token;
    if (token.find("subgraph_") != std::string::npos) {
      curr_subgraph=token;
      continue;
    }
    ss2 >> id;
    if (op_id_.size() <= static_cast<size_t>(id)) {
      op_id_.resize(id+1);
      data_entry_.resize(id+1);
    }
    int64_t total_elements=1;
    std::vector<int64_t> shape;
    if (token == "input") {
      int64_t size=0;
      while (ss2 >> size) {
        total_elements *= size;
        shape.push_back(size);
      }
    } else {
      op_id_[id]=token; 
// 注1
      bool shape_data=false;
      NodeEntry entry;
      while (ss2 >> token) {
        if (token == "shape:") {
          shape_data=true;
        } else if (shape_data) {
          total_elements *= std::stoll(token);
          shape.push_back(std::stoll(token));
        } else if (token != "inputs:") {
          entry.inputs.push_back(std::stoi(token));
        }
      }
      entry.id=id;
      entry.output=id;
      graph_[curr_subgraph].push_back(entry); 
// 注2
    }
    DLDevice dev;
    dev.device_type=static_cast<DLDeviceType>(1);
    dev.device_id=0;
data_entry_[id]=NDArray::Empty(shape, DLDataType{kDLFloat, 32, 1}, dev); 
// 注3
  }
}
8. 实现 GetFunction到TVM运行时分析
实现GetFunction为TVM运行时提供可执行的子图函数。实现代码如下所示：
PackedFunc GetFunction(const std::string& name, 
                       const ObjectPtr<Object>& sptr_to_self) final {
  if (this->graph_.find(name) != this->graph_.end()) {
    this->curr_subgraph_=name;
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {

      // 将输入张量复制到相应的数据条目
      for (auto i=0; i < args.size(); ++i) {
        ICHECK(args[i].type_code() == kNDArrayContainer || args[i].type_code() == kArrayHandle)
            << "Expect NDArray or DLTensor as inputs\n";
        if (args[i].type_code() == kArrayHandle) {
          DLTensor* arg=args[i];
          this->data_entry_[i].CopyFrom(arg);
        } else {
          NDArray arg=args[i];
          this->data_entry_[i].CopyFrom(arg);
        }
      }
      // 执行子图
      for (const auto& it : this->graph_[this->curr_subgraph_]) {
        this->Run(it.id, it.inputs, it.output);
      }
      ICHECK_GT(graph_.count(this->curr_subgraph_), 0U);
      // 将数据项的输出复制到TVM运行时参数
      auto out_idx=graph_[this->curr_subgraph_].back().output;
      if (args[args.size() - 1].type_code() == kArrayHandle) {
        DLTensor* arg=args[args.size() - 1];
        this->data_entry_[out_idx].CopyTo(arg);
      } else {
        NDArray arg=args[args.size() - 1];
        this->data_entry_[out_idx].CopyTo(arg);
      }
      *rv=data_entry_.back();
    });
  } else {
    LOG(FATAL) << "Unknown subgraph: " << name << "\n";
    return PackedFunc();
  }
}
9. 实施运行，存储及函数注册
实现代码如下所示：
void Run(int id, const std::vector<int>& inputs, int output) {
  // 列出数据输入索引
  std::vector<int> args(inputs.begin(), inputs.end());
  args.push_back(output);

  // 初始化数据保持器
  std::vector<TVMValue> values(args.size());
  std::vector<int> type_codes(args.size());

  // 使用TVMValue及类型代码，初始化TVM arg setter
  TVMArgsSetter setter(values.data(), type_codes.data());

  // 将每个参数设置为相应的数据项
  if (op_id_[id] == "add" || op_id_[id] == "sub" || op_id_[id] == "mul") {
    for (size_t i=0; i < args.size(); i++) {
      setter(i, data_entry_[args[i]]);
    }
  }
  // 调用相应的运算符函数
  if (op_id_[id] == "add") {
    Add(values.data(), type_codes.data(), args.size());
  } else if (op_id_[id] == "sub") {
    Sub(values.data(), type_codes.data(), args.size());
  } else if (op_id_[id] == "mul") {
    Mul(values.data(), type_codes.data(), args.size());
  } else {
    LOG(FATAL) << "Unknown op: " << op_id_[id] << "\n";
  }
}
代码如下所示：
TVM_REGISTER_GLOBAL("module.examplejson_module_create")
.set_body_typed([](std::string code){
    auto n=make_object<ExampleJsonModule>(code);
    return runtime::Module(n);
});
2）实现 SaveToBinary与LoadFromBinary
代码如下所示：
void SaveToBinary(dmlc::Stream* stream) final {
    stream->Write(this->graph_json_);
}
同理，LoadFromBinary读取子图流并重新构建自定义的运行时模块。实现代码如下所示：
static Module LoadFromBinary(void* strm) {
  dmlc::Stream* stream=static_cast<dmlc::Stream*>(strm);
  std::string graph_json;
  stream->Read(&graph_json);
  auto n=tvm::runtime::make_object<ExampleJsonModule>(graph_json);
  return Module(n);
}
需要注册此函数，启用相应的Python API，代码如下所示：
TVM_REGISTER_GLOBAL("module.loadbinary_examplejson")
.set_body_typed(ExampleJsonModule::LoadFromBinary);
上面的注册意味着当用户调用tvm.runtime.load(lib_path)API导出的库，具有ExampleJSON流时，LoadFromBinary调用创建相同的自定义运行时模块。
如果想直接从ExampleJSON文件支持模块创建，可以实现一个简单的函数并注册Python API，代码如下所示：
static Module Create(const std::string& path) {
    std::ifstream filep;
    filep.open(path, std::ios::in);
    std::string graph_json;
    std::string line;
    while (std::getline(filep, line)) {
        graph_json += line;
        graph_json += "\n";
    }
    filep.close();
    auto n=tvm::runtime::make_object<ExampleJsonModule>(graph_json);
    return Module(n);
}
TVM_REGISTER_GLOBAL("module.loadfile_examplejson")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv=ExampleJsonModule::Create(args[0]);
});
用户可以手动编写/修改ExampleJSON文件，使用Python API tvm.runtime.load("mysubgraph.examplejson", "examplejson")构造自定义模块。代码如下所示：
tvm.runtime.load_module("mysubgraph.examplejson", "examplejson")
7.4.4　Runtime运行时机制分析
1. 将DNNL（深度神经网络库）集成到TVM：注释规则
实现代码如下所示：
def _register_external_op_helper(op_name, supported=True):
    @tvm.ir.register_op_attr(op_name, "target.dnnl")
    def _func_wrapper(attrs, args):
        return supported
    return _func_wrapper
 
_register_external_op_helper("nn.batch_norm")
_register_external_op_helper("nn.conv2dConv2d")
_register_external_op_helper("nn.dense")
_register_external_op_helper("nn.relu")
_register_external_op_helper("add")
_register_external_op_helper("subtract")
_register_external_op_helper("multiply")
2. 图形模式的规则
可以将DNNL以下代码片段实现如下所示：
DNNLConv2dConv2d(const bool has_bias=false, const bool has_relu=false) {
  // 创建卷积操作描述... skip ...
  auto conv_desc=dnnl::convolution_forward::desc(
    dnnl::prop_kind::forward_inference, 
    dnnl::algorithm::convolution_direct, 
    conv_src_md, conv_weights_md, conv_bias_md, conv_dst_md, 
    strides_dims, padding_dims_l, padding_dims_r);
 
  // Attach 附加ReLU
  dnnl::primitive_attr attr;
  if (has_relu) {
    dnnl::post_ops ops;
    ops.append_eltwise(1.f, dnnl::algorithm::eltwise_relu, 0.f, 0.f);
    attr.set_post_ops(ops);
  }
  auto conv2dConv2d_prim_desc=dnnl::convolution_forward::primitive_desc(
    conv_desc, attr, engine_);
  // 创建卷积操作原语描述... skip ...
在本例中，除了单个conv2dConv2d，希望将图形模式conv2dConv2d+Rrelu映射到DNNLConv2dConv2d（false，true），将conv2dConv2d+Aadd+Rrelu映射到DNNLConv2dConv2d（true，true）。用下面的代码片段实现：
def make_pattern(with_bias=True):
  data=wildcard()
  weight=wildcard()
  bias=wildcard()
  conv=is_op('nn.conv2dConv2d')(data, weight)
  if with_bias:
    conv_out=is_op('add')(conv, bias)
  else:
    conv_out=conv
  return is_op('nn.relu')(conv_out)
 
@register_pattern_table("dnnl")
def pattern_table():
  conv2dConv2d_bias_relu_pat=("dnnl.conv2dConv2d_bias_relu", make_pattern(with_bias=True))
  conv2dConv2d_relu_pat=("dnnl.conv2dConv2d_relu", make_pattern(with_bias=False))
  dnnl_patterns=[conv2dConv2d_bias_relu_pat, conv2dConv2d_relu_pat]
  return dnnl_patterns
在DNNL示例中，实现了两个具有不同名称的模式，可以在codegenCodeGen中轻松地识别。这些模式是用Relay模式语言实现的。可以学习如何编写模式。
通过模式表，可以使用一个Relay pass执行。实现代码如下所示：
%1=nn.conv2dConv2d(%data, %weight, ...)
%2=add(%1, %bias)
%3=nn.relu(%2)
to
%1=fn(%input1, %input2, %input3, 
        Composite="dnnl.conv2dConv2d_bias_relu", 
        PartitionedFromPattern="nn.conv2dConv2d_add_nn.relu_") {
  %1=nn.conv2dConv2d(%input1, %input2, ...)
  %2=add(%1, %input3)
  nn.relu(%2)
}
%2=%1(%data, %weight, %bias)

实现代码如下所示:
def make_pattern(with_bias=True):
  data=wildcard()
  weight=wildcard()
  conv=is_op('nn.conv2dConv2d')(data, weight)
  return wildcard()(conv)
使用注释规则，可用BYOC Relay pass列表转换Relay图，
实现代码如下所示：
mod=create_relay_module_from_model() # Output: Figure 1
mod=transform.MergeComposite(pattern_table)(mod)
mod=transform.AnnotateTarget([“dnnl”])(mod) # Output: Figure 2
mod=transform.MergeCompilerRegions()(mod) # Output: Figure 3
mod=transform.PartitionGraph()(mod) # Output: Figure 4
7.5　在TVM上集成部署CodegenCodeGen示例分析
7.5.1　 CodegenCodeGen与Runtime转化成TVM
实现代码如下所示：
runtime::Module DNNLCompiler(const ObjectRef& ref) {
  //  "ref"会将分割成Relay功能，满足kCompiler=dnnl
  CHECK(ref->IsInstance<FunctionNode>());
  auto func=Downcast<Function>(ref);
  // 获取函数名作为要在运行时匹配的符号
  auto func_name=GetExtSymbol(func);
  // 将函数序列化为JSON字符串
  DNNLJSONSerializer serializer(func_name, func);
  serializer.serialize();
  std::string graph_json=serializer.GetJSON();
  // 已绑定到模块的常量张量名称
  // 调用export_library时，所有常量张量将与JSON图一起序列化
  auto params=serializer.GetParams();
  // 创建DNNL JSON运行时的函数
  const auto* pf=runtime::Registry::Get("runtime.DNNLJSONRuntimeCreate");
  CHECK(pf != nullptr) << "Cannot find JSON runtime module to create";
  // 创建可以运行序列化函数的DNNL运行时模块
  auto mod=(*pf)(func_name, graph_json, params);
  return mod;
}
TVM_REGISTER_GLOBAL("relay.ext.dnnl").set_body_typed(DNNLCompiler);
runtime负责Relay函数，在single .so中有多个DNNL 运行时模块。
1.实现DNNL JSON序列化
实现代码如下所示：
{
  op: "kernel", 
  name: "dnnl.conv2dConv2d_relu", 
  inputs: [[0, 0, 0], [1, 0, 0]], 
  attrs: {
    PartitionedFromPattern: ["nn.conv2dConv2d_nn.relu_"], 
    shape: [1, 32, 14, 14]
  }
}
在runtime仍然需要Conv2DConv2d属性，比如padding和stripes，但是BYOC JSON序列化程序，只附加复合函数的属性，不附加body实现算子。另一方面，定制的DNNL JSON序列化程序，在复合函数中附加第一个，也是唯一一个Conv2DConv2d的属性，生成以下JSON节点。实现代码如下所示：
{
  op: "kernel", 
  name: "dnnl.conv2dConv2d_relu", 
  inputs: [[0, 0, 0], [1, 0, 0]], 
  attrs: {
    shape: [1, 32, 14, 14], 
    data_layout: ["NCHW"], 
    kernel_layout: ["OIHW"], 
    strides: [1, 1], 
    padding: [1, 1, 1, 1]
  }
}
从DNNL JSON序列化程序，只要JSON runtime能够解释，就可以定制序列化程序，生成JSON格式的任何表单。
2.构建DNNL JSON 运行时
实现DNNL JSON 运行时，解释与执行序列化图。放在src/runtime/contrib/dnnl/dnnl_json_runtime.cc中。
首先注册两个运行时 API。执行序列化runtime.DNNLJSONRuntimeCreate后，runtime.module.loadbinary_dnnl_json可在 load.so 中使用。实现代码如下所示：
// 创建DNNL JSON运行时，以便解释和执行给定的JSON图
runtime::Module DNNLJSONRuntimeCreate(String symbol_name, String graph_json,  const Array<String>& const_names) {
  auto n=make_object<DNNLJSONRuntime>(symbol_name, graph_json, const_names);
  return runtime::Module(n);
}
TVM_REGISTER_GLOBAL("runtime.DNNLJSONRuntimeCreate")
    .set_body_typed(DNNLJSONRuntimeCreate);
TVM_REGISTER_GLOBAL("runtime.module.loadbinary_dnnl_json")
    .set_body_typed(JSONRuntimeBase::LoadFromBinary<DNNLJSONRuntime>);
// 解释DNNL JSON运行时实现。基本类结构是：
class DNNLJSONRuntime : public JSONRuntimeBase {
  const  char* type_key() const { return  "dnnl_json"; } 
  void Init(const Array<NDArray>& consts) override {
    // 初始化DNNL图形引擎
    BuildEngine();
    
    // 设置权重常量
    CHECK_EQ(consts.size(), const_idx_.size())
      << "The number of input constants must match the number of required.";
    SetupConstants(consts);
  }
 
void Run() override {
   // 1. 写输入缓冲区
   // 2. 通过绘制流来调用引擎
   // 3. 读取和填充输出缓冲区
  }
}
3.用CodegenCodeGen将DNNL生成TVM
首先，使用TVM注册API注册代码源。该注册使TVM编译引擎使用Compiler=<your codegenCodeGen> 来分发Relay功能relay.ext.<your codegenCodeGen>。然后，实现DNNL编译器的入口函数。代码如下所示：
runtime::Module DNNLCompiler(const ObjectRef& ref) {
  DNNLModuleCodegenCodeGen dnnl;
  return dnnl.CreateCSourceModule(ref);
}
TVM_REGISTER_GLOBAL("relay.ext.dnnl").set_body_typed(DNNLCompiler);
运行时负责Relay，而在single .so中有多个DNNL 运行时。
代码如下所示：
runtime::Module CreateCSourceModule(const ObjectRef& ref) override {
    // 包含头文件Include headers
    // ...skip...
    code_stream_ << "#include <dnnl/dnnl_kernel.h>\n";
    // ...skip...
 
    // "ref"将会是kCompiler=dnnl的分区Relay功能
    CHECK(ref->IsInstance<FunctionNode>());
    auto res=GenDNNLFunc(Downcast<Function>(ref));
    // "code" 是使用DNNL API生成的C代码
    std::string code=code_stream_.str();
    // "res"是常数权重（符号、值）的元组
    // 调用export_library时，所有常量张量将与生成的C代码一起序列化
    String sym=std::get<0>(res);
    Array<String> variables=std::get<1>(res);
    // 创建一个包含所有上述特性的CSource模块
    const auto* pf=runtime::Registry::Get("runtime.CSourceModuleCreate");
    CHECK(pf != nullptr) << "Cannot find csource module to create the external runtime module";
    return (*pf)(code, "c", sym, variables);
  }
参阅嵌入的注释，以获取与TVM C源Runtime模块兼容的功能接口的说明。实现代码如下所示：
// 示例Relay图: conv2dConv2d -> Aadd -> Rrelu.
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/container.h>
#include <tvm/runtime/packed_func.h>
#include <dlpack/dlpack.h>
#include <dnnl/dnnl_kernel.h>
using namespace tvm::runtime;
using namespace tvm::runtime::contrib;
// 使用DNNL执行conv2dConv2d->add->relu图
extern "C" void dnnl_0_(float* dnnl_0_i0, float* dnnl_0_i1, 
              float* dnnl_0_i2,float* out0) 
{
  // 分配中间缓冲区
  float* buf_0=(float*)std::malloc(4 * 4608);
  float* buf_1=(float*)std::malloc(4 * 4608);
  float* buf_2=(float*)std::malloc(4 * 4608);
// 预实现算子DNNL函数预实现的基于op的DNNL函数
  dnnl_conv2dConv2d(dnnl_0_i0, dnnl_0_i1, buf_0, 1, 32, 14, 14, 32, 1, 0, 0, 3, 3, 1, 1);
  dnnl_add(buf_0, dnnl_0_i2, buf_1, 1, 32, 12, 12);
  dnnl_relu(buf_1, buf_2, 1, 32, 12, 12);
// 将最终输出复制到相应的缓冲区
  std::memcpy(out0, buf_2, 4 * 4608);
  std::free(buf_0);
  std::free(buf_1);
  std::free(buf_2);
}
// 具有DLTensor类型的所有参数的包装函数
extern "C" int dnnl_0_wrapper_(DLTensor* arg0, 
        DLTensor* arg1, 
        DLTensor* arg2, 
        DLTensor* out0) {
// 将所有DLTensor强制转换为基元类型缓冲区并调用上述可执行函数
  dnnl_0_(static_cast<float*>(arg0->data), 
  static_cast<float*>(arg1->data), 
  static_cast<float*>(arg2->data), 
  static_cast<float*>(out0->data));
  return 0;
}
// TVM宏用“dnnl_0_wrapper_”生成TVM运行时兼容函数“dnnl_0”
TVM_DLL_EXPORT_TYPED_FUNC(dnnl_0, dnnl_0_wrapper_);
// 实现DNNL函数src/runtime/contrib/dnnl/dnnl.cc。
rest实现在src/relay/backend/contrib/dnnl/codegenCodeGen.cc中，生成TVM运行时co
degen。 
4.用库编译生成C代码
生成DNNLCompiler的C代码，而GCC没有编译为二进制。可用export_libray(mod)编译生成C代码，实现代码如下所示：
def update_lib(lib):
    # 包括src/runtime/contrib/dnnl/dnnl.cc的路径
    test_dir=os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
    source_dir=os.path.join(test_dir, "..", "..", "..")
    contrib_path=os.path.join(source_dir, "src", "runtime", "contrib")
 
    # 设置gcc标志以编译DNNL代码
    kwargs={}
    kwargs["options"]=["-O2", "-std=C++14", "-I"+contrib_path]
    tmp_path=util.tempdir()
    lib_name='lib.so'
    lib_path=tmp_path.relpath(lib_name)
    # 使用DNNL API生成的C代码编译为二进制lib.so
    lib.export_library(lib_path, fcompile=False, **kwargs)
    # 加载lib.so回到运行时模块
    lib=runtime.load_module(lib_path)
    return lib
 
with tvm.transform.PassContext(opt_level=3):
    json, lib, param=relay.build(mod, target=target, params=params)
lib=update_lib(lib)
rt_mod=tvm.contrib.graph_runtime.create(json, lib, ctx)
7.6　Code Generation代码生成应用实践
7.6.1　表达式编译
2. 解析执行的缺陷
代码如下所示：
std::ostringstream macro_stream;
std::ostringstream decl_stream;
std::ostringstream buf_stream;
实现代码如下所示：
@Override public Object visitPlus(CalculatorParser.PlusContext ctx) {
    Object left = visit(ctx.plusOrMinus());
    Object right = visit(ctx.multOrDiv());
    if (left instanceof Long && right instanceof Long) {
        return (Long) left + (Long) right;
    } else if (left instanceof Long && right instanceof Double) {
        return (Long) left + (Double) right;
    } else if (left instanceof Double && right instanceof Long) {
        return (Double) left + (Long) right;
    } else if (left instanceof Double && right instanceof Double) {
        return (Double) left + (Double) right;
    }
    throw new IllegalArgumentException();
}
3. 推理验证
实现代码如下所示：
@Override public AlgebraNode visitPlus(CalculatorParser.PlusContext ctx) {
    return new PlusNode(visit(ctx.plusOrMinus()), visit(ctx.multOrDiv()));
}
AlgebraNode 接口定义：
public interface AlgebraNode {
    DataType getType(); // Validate 和 CodeGenCodeGen 都会用到
    String generateCode(); // CodeGenCodeGen 使用
    List<AlgebraNode> getInputs();
}
实现类与 AST节点相对应。
对于加法，类型推导的过程很简单——如果两个操作数都是 Long，结果为 Long，否则为 Double。实现代码如下所示：
@Override public DataType getType() {
    if (dataType == null) {
        dataType = inferTypeFromInputs();
    }
    return dataType;
}

private DataType inferTypeFromInputs() {
    for (AlgebraNode input : getInputs()) {
        if (input.getType() == DataType.DOUBLE) {
            return DataType.DOUBLE;
        }
    }
    return DataType.LONG;
}
4. 生成加法强类型代码
可将加法生成强类型代码如下所示：
@Override public String generateCode() {
    if (getLeft().getType() == DataType.DOUBLE && getRight().getType() == DataType.DOUBLE) {
        return "(" + getLeft().generateCode() + " + " + getRight().generateCode() + ")";
    } else if (getLeft().getType() == DataType.DOUBLE && getRight().getType() == DataType.LONG) {
        return "(" + getLeft().generateCode() + " + (double)" + getRight().generateCode() + ")";
    } else if (getLeft().getType() == DataType.LONG && getRight().getType() == DataType.DOUBLE) {
        return "((double)" + getLeft().generateCode() + " + " + getRight().generateCode() + ")";
    } else if (getLeft().getType() == DataType.LONG && getRight().getType() == DataType.LONG) {
        return "(" + getLeft().generateCode() + " + " + getRight().generateCode() + ")";
    }
    throw new IllegalStateException();
}
7.6.2　编译IRModule方案
1. 编译并运行 IRModule
调用tvm.build函数，可对 IRModule 直接编译，然后运行查看结果是否符合预期。实现代码如下所示：
import numpy as np

mod = tvm.build(ir_module, target="c")
# mod = tvm.build(ir_module, target="llvm")
# mod = tvm.build(ir_module, target="cuda")

a = tvm.nd.array(np.arange(8).astype("float32"))
print(a)
# [0. 1. 2. 3. 4. 5. 6. 7.]

b = tvm.nd.array(np.zeros((8,)).astype("float32"))
mod(a, b)
print(b)
# [1. 2. 3. 4. 5. 6. 7. 8.]
2. 根据目标调用相应的 CodeGen
实现代码如下所示：
// src/target/source/codegen_cCodeGenC_host.cc
TVM_REGISTER_GLOBAL("target.build.c").set_body_typed(BuildCHost);

// src/target/opt/build_cuda_on.cc
TVM_REGISTER_GLOBAL("target.build.cuda").set_body_typed(BuildCUDA);

// src/target/llvm/llvm_module.cc
TVM_REGISTER_GLOBAL("target.build.llvm")
    .set_body_typed([](IRModule mod, Target target) -> runtime::Module {
      auto n = make_object<LLVMModuleNode>();
      n->Init(mod, target);
      return runtime::Module(n);
    });
1）target="C"
如果 target="c"，那么tvm.build最终调用的是提前注册的 target.build.c 的全局函数，并且位于源代码 tvm/src/target/source/codegen_cCodeGenC_host.cc中（省略了部分辅助代码），如下所示：
runtime::Module BuildCHost(IRModule mod, Target target) {
  // Step 1: Initialize初始化C++中 CodeGenCodeGen for C++
  CodeGenCodeGenCHost cg;
  cg.Init(output_ssa, emit_asserts, target->str(), devices);

  // Step 2: 将IRModule中的所有Add all tir::PrimFunc 添加到编译列表中in IRModule to compile list
  for (auto& kv : funcs) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>()) << "CodegenCodeGenCHost: Can only take PrimFunc";
    auto f = Downcast<PrimFunc>(kv.second);
    cg.AddFunction(f);
  }
  
  // Step 3: 将IRModule下译到C++源代码Lower IRModule to C++ source code
  std::string code = cg.Finish();

  // Step 4: 编译C++源代码并创建Compile C++ source code and create runtime::Module封装 wrapper
  return CSourceModuleCreate(code, "c", cg.GetFunctionNames());
}
2）target="cuda"
如果 target="cuda"，tvm.build最终调用的是提前注册的 target.build.cuda 的全局函数，并且位于源代码 tvm/src/target/opt/build_cuda_on.cc中，实现如下所示：
runtime::Module BuildCUDA(IRModule mod, Target target)
  // Step 1: Initialize初始化CUDA中 CodeGenCodeGen for CUDA  
  bool output_ssa = false;
  CodeGenCodeGenCUDA cg;
  cg.Init(output_ssa);
  
  // Step 2: 将IRModule中的所有tir::PrimFunc添加到编译列表中Add all tir::PrimFunc in IRModule to compile list
  for (auto kv : mod->functions) {
    auto f = Downcast<PrimFunc>(kv.second);
    cg.AddFunction(f);
  }

  // Step 3: 将IRModule下译到CUDA源代码Lower IRModule to CUDA source code
  std::string code = cg.Finish();

  // Step 4: 使用NVCC编译CUDA源代码并创建Compile CUDA source code using NVCC and create runtime::Module
  std::string fmt = "ptx";
  std::string ptx = NVRTCCompile(code, cg.need_include_path());
  return CUDAModuleCreate(ptx, fmt, ExtractFuncInfo(mod), code);
}
3）target="llvm"
实现代码如下所示：
void Init(const IRModule& mod, const Target& target) {
  // Step 1: 初始化具有不同目标的LLVM的CodeGenInitialize CodeGen for LLVM with different target
  InitializeLLVM();
  tm_ = GetLLVMTargetMachine(target);
  std::unique_ptr<CodeGenCodeGenLLVM> cg = CodeGenCodeGenLLVM::Create(tm_.get());

  // Step 2: 将IRModule中的所有tir::PrimFunc添加到编译列表中Add all tir::PrimFunc in IRModule to compile list
  std::vector<PrimFunc> funcs;
  for (auto kv : mod->functions) {
    if (!kv.second->IsInstance<PrimFuncNode>()) {
      // (@jroesch): we relax constraints here, Relay functions will just be ignored.
      DLOG(INFO) << "Can only lower IR ModuleIRModule with PrimFuncs, but got "
                  << kv.second->GetTypeKey();
      continue;
    }
    auto f = Downcast<PrimFunc>(kv.second);
    funcs.push_back(f);
  }

  // Step 3: 下译IRModule至LLVM IR代码Lower IRModule to LLVM IR code
  module_ = cg->Finish();
}
7.6.3　CodeGenCodeGen 原理：以 CodeGenCodeGenC为例
实现代码如下所示：
class CodeGenCodeGenC: public ExprFunctor<void(const PrimExpr&, std::ostream&)>,
                public StmtFunctor<void(const Stmt&)>,
                public CodeGenCodeGenSourceBase {
 public:
  // expression
  void VisitExpr_(const VarNode* op, std::ostream& os) override;         
// NOLINT(*)
  void VisitExpr_(const LoadNode* op, std::ostream& os) override;        
// NOLINT(*)
  void VisitExpr_(const BufferLoadNode* op, std::ostream& os) override;  
// NOLINT(*)
  void VisitExpr_(const LetNode* op, std::ostream& os) override;         
// NOLINT(*)
  void VisitExpr_(const CallNode* op, std::ostream& os) override;        
// NOLINT(*)
  void VisitExpr_(const AddNode* op, std::ostream& os) override;         
// NOLINT(*)
  void VisitExpr_(const SubNode* op, std::ostream& os) override;         
// NOLINT(*)
  void VisitExpr_(const MulNode* op, std::ostream& os) override;         
// NOLINT(*)
  void VisitExpr_(const DivNode* op, std::ostream& os) override;         
// NOLINT(*)
  void VisitExpr_(const ModNode* op, std::ostream& os) override;         
// NOLINT(*)
  void VisitExpr_(const MinNode* op, std::ostream& os) override;         
// NOLINT(*)
  void VisitExpr_(const MaxNode* op, std::ostream& os) override;         
// NOLINT(*)
  void VisitExpr_(const EQNode* op, std::ostream& os) override;          
// NOLINT(*)
  void VisitExpr_(const NENode* op, std::ostream& os) override;          
// NOLINT(*)
  void VisitExpr_(const LTNode* op, std::ostream& os) override;          
// NOLINT(*)
  void VisitExpr_(const LENode* op, std::ostream& os) override;          
// NOLINT(*)
  void VisitExpr_(const GTNode* op, std::ostream& os) override;          
// NOLINT(*)
  void VisitExpr_(const GENode* op, std::ostream& os) override;          
// NOLINT(*)
  void VisitExpr_(const AndNode* op, std::ostream& os) override;         
// NOLINT(*)
  void VisitExpr_(const OrNode* op, std::ostream& os) override;          
// NOLINT(*)
  void VisitExpr_(const CastNode* op, std::ostream& os) override;        
// NOLINT(*)
  void VisitExpr_(const NotNode* op, std::ostream& os) override;         
// NOLINT(*)
  void VisitExpr_(const SelectNode* op, std::ostream& os) override;      
// NOLINT(*)
  void VisitExpr_(const RampNode* op, std::ostream& os) override;        
// NOLINT(*)
  void VisitExpr_(const ShuffleNode* op, std::ostream& os) override;     
// NOLINT(*)
  void VisitExpr_(const BroadcastNode* op, std::ostream& os) override;   
// NOLINT(*)
  void VisitExpr_(const IntImmNode* op, std::ostream& os) override;      
// NOLINT(*)
  void VisitExpr_(const FloatImmNode* op, std::ostream& os) override;    
// NOLINT(*)
  void VisitExpr_(const StringImmNode* op, std::ostream& os) override;   
// NOLINT(*)
  // statment
  void VisitStmt_(const LetStmtNode* op) override;
  void VisitStmt_(const StoreNode* op) override;
  void VisitStmt_(const BufferStoreNode* op) override;
  void VisitStmt_(const ForNode* op) override;
  void VisitStmt_(const WhileNode* op) override;
  void VisitStmt_(const IfThenElseNode* op) override;
  void VisitStmt_(const AllocateNode* op) override;
  void VisitStmt_(const AttrStmtNode* op) override;
  void VisitStmt_(const AssertStmtNode* op) override;
  void VisitStmt_(const EvaluateNode* op) override;
  void VisitStmt_(const SeqStmtNode* op) override;
  void VisitStmt_(const AllocateConstNode* op) override;
}
可以看到，CodeGenCodeGenC 会遍历到两种 TIR Node：Expression（表达式） 和 Statement（语句）。Expression（表达式）中包含了常见的变量声明、运算、判断、函数调用。Statement（语句）中包含了控制流（if-else，Loop 等）、内存管理、赋值等操作。
例如，遇到四则运算的 Expression，可用CodeGenCodeGenC 直接翻译为 " a OP b "的代码，实现如下所示：
template <typename T>
inline void PrintBinaryExpr(const T* op, const char* opstr,
                            std::ostream& os, CodeGenCodeGenC* p) {
  // If both a and b are scalars
  if (op->dtype.lanes() == 1) {
    // If OP is an alphabet string, then lower it as "OP(a, b)"
    if (isalpha(opstr[0])) {
      os << opstr << '(';
      p->PrintExpr(op->a, os);
      os << ", ";
      p->PrintExpr(op->b, os);
      os << ')';
    }
    // If OP is a symbol, like + - * / %, then lower it as "a OP b"
    else {
      os << '(';
      p->PrintExpr(op->a, os);
      os << ' ' << opstr << ' ';
      p->PrintExpr(op->b, os);
      os << ')';
    }
  }
  // If both a and b are vectors
  else {
    p->PrintVecBinaryOp(opstr, op->dtype, op->a, op->b, os);
  }
}

void CodeGenCodeGenC::VisitExpr_(const AddNode* op, std::ostream& os) {  
// NOLINT(*)
  PrintBinaryExpr(op, "+", os, this);
}
void CodeGenCodeGenC::VisitExpr_(const SubNode* op, std::ostream& os) {  
// NOLINT(*)
  PrintBinaryExpr(op, "-", os, this);
}
void CodeGenCodeGenC::VisitExpr_(const MulNode* op, std::ostream& os) {  
// NOLINT(*)
  PrintBinaryExpr(op, "*", os, this);
}
void CodeGenCodeGenC::VisitExpr_(const DivNode* op, std::ostream& os) {  
// NOLINT(*)
  PrintBinaryExpr(op, "/", os, this);
}
如果选择 SelectNode，CodeGenC 将 "(c ? a : b)" 翻译为如下所示代码如果遇到选择 SelectNode，使用CodeGenC 则翻译为 "(c ? a : b)" 的代码：
void CodeGenCodeGenC::VisitExpr_(const SelectNode* op, std::ostream& os) {
  os << "(";
  PrintExpr(op->condition, os);
  os << " ? ";
  PrintExpr(op->true_value, os);
  os << " : ";
  PrintExpr(op->false_value, os);
  os << ")";
}
如果选择 SelectNode，CodeGenC 将 "(c ? a : b)" 翻译为如下所示代码如果遇到选择 SelectNode，CodeGenC 则翻译为 "(c ? a : b)" 的代码翻译如下所示：
for (DTYPE VID = 0; VID < EXTEND; ++VID) {
BODY
}\n
在CodeGen过程中，实际上是在遍历tir Stmt的AST，因为生成的循环都是基于For的，调用过程也比较简单了。过程代码如下所示代码如下所示：
void CodeGenCodeGenC::VisitStmt_(const ForNode* op) {
  std::string extent = PrintExpr(op->extent);
  PrintIndent();
  std::string vid = AllocVarID(op->loop_var.get());
  ICHECK(is_zero(op->min));
  stream << "for (";
  PrintType(op->loop_var.dtype(), stream);
  stream << ' ' << vid << " = 0; " << vid << " < " << extent << "; ++" << vid << ") {\n";
  int for_scope = BeginScope();
  PrintStmt(op->body);
  this->EndScope(for_scope);
  PrintIndent();
  stream << "}\n";
}
7.7　CodeGen调用关系工程示例分析
实现代码如下所示：
LoweredOutput CodegenCodeGen(IRModule mod, relay::Function func, String mod_name) {
    mod_name_ = mod_name;
    VLOG_CONTEXT << "GraphExecutorCodegenCodeGen";
    VLOG(1) << "compiling:" << std::endl << PrettyPrint(func);

    // TODO:为什么要在降低内存和更新工作空间大小之前进行调度？
    memory_plan_ = GraphPlanMemory(func);

    backend::FunctionInfo func_info;

    if (memory_plan_.defined()) {
      // TODO: remove移除 UpdateMainWorkspaceSize
      func_info =
          relay::tec::UpdateMainWorkspaceSize(mod, config_, memory_plan_->expr_to_storage_info);
      mod = WithAttr(mod, "main_func_info", func_info);
    }

    IRModule lowered_mod = tec::LowerTE(mod_name_, config_, [this](BaseFunc func) {
      // 需要保持外部函数的恒定映射，因此传递这个处理函数，允许在降级优化下译优化每个函数时处理
      if (func->GetAttr<String>(attr::kCompiler).defined()) {
        UpdateConstants(func, &params_);
      }

      // TODO: 应该重构以作为进一步的传递来执行，而不是将数据写入
         201,3,25%
      // 直接降级优化下译优化过程
      tec::UpdateFunctionMetadata(func, this->function_metadata_);
    })(mod);

    Optional<backend::FunctionInfo> main_func_info =
        lowered_mod->GetAttr<backend::FunctionInfo>("main_func_info");

    function_metadata_.Set(runtime::symbol::tvm_module_main, main_func_info.value());

    Function lowered_main_func = Downcast<Function>(lowered_mod->Lookup("main"));
// 现在已经将所有运算符降级优化下译优化到TIR代码，可以继续编译了。
// 需要重新规划，完成下译优化重构功能因为之前的结果因降级优化将在未来的重构中解决这个问题。
    memory_plan_ = GraphPlanMemory(lowered_main_func);
    // 图形调度器也不能处理对全局变量的规划调用，必须重新映射
    // 首先，将所有参数转换为输入节点。
    for (auto param : lowered_main_func->params) {
      auto node_ptr = GraphInputNode::make_node_ptr(param->name_hint(), GraphAttrs());
      var_map_[param.get()] = AddNode(node_ptr, param);
    }

    heads_ = VisitExpr(lowered_main_func->body);
    std::ostringstream os;
    dmlc::JSONWriter writer(&os);
    GetJSON(&writer);
    LoweredOutput ret;
    ret.graph_json = os.str();

    // 收集由外部codegenCodeGen生成的任何运行时模块
    ret.external_mods =
        lowered_mod->GetAttr<Array<runtime::Module>>(tvm::attr::kExternalMods).value_or({});

    // 收集由外部codegenCodeGen提取的任何常数
    ret.params = std::unordered_map<std::string, tvm::runtime::NDArray>();
    Map<String, runtime::NDArray> const_name_to_constant =
        lowered_mod->GetAttr<Map<String, runtime::NDArray>>(tvm::attr::kConstNameToConstant)
            .value_or({});
    for (const auto& kv : const_name_to_constant) {
      VLOG(1) << "constant '" << kv.first << "' contributed by external codegenCodeGen";
      ICHECK(ret.params.emplace(kv.first, kv.second).second);
    }

    // 收集降级优化下译优化过程中提取的任何常数
    for (const auto& kv : params_) {
      VLOG(1) << "constant '" << kv.first << "' contributed by TECompiler";
      ICHECK(ret.params.emplace(kv.first, kv.second).second);
    }

    ret.function_metadata = std::move(function_metadata_);

    // 这就是按目标分离模块中功能点划分的
    ret.lowered_funcs = tec::GetPerTargetModules(lowered_mod);
    ret.metadata =
        ExecutorCodegenCodeGenMetadata(
{} /* inputs */, {} /* input_tensor_types */, {} /* outputs */,
{} /* output_tensor_types */, {} /* pools */, {} /* devices */,
runtime::kTvmExecutorGraph /* executor */, mod_name_ /* mod_name */,
"packed" /* interface_api */, Bool(false) /* unpacked_api */);
    return ret;
  }

 
第8章　后端部署与OpenCL
8.1　OpenCL概述与开发示例
8.3　OpenCL构建编程示例分析
8.3.6　读取结果并释放OpenCL资源
主程序实现如下所示：
#include <iostream>
#include <fstream>
#include <sstream>
#include <CL/cl.h>
const int ARRAY_SIZE = 1000;
// 选择OpenCL平台，创建上下文
cl_context CreateContext()
{
	cl_int errNum;
	cl_uint numPlatforms;
	cl_platform_id firstPlatformId;
	cl_context context = NULL;
	// 选择平台中第一个
	errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
	if (errNum != CL_SUCCESS || numPlatforms <= 0)
	{
		std::cerr << "Failed to find any OpenCL platforms." << std::endl;
		return NULL;
	}
	// 创建OpenCL上下文环境
	cl_context_properties contextProperties[] =
	{
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)firstPlatformId,
		0
	};
	context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
		NULL, NULL, &errNum);
	return context;
}
// 创建设备及命令队列
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device)
{
	cl_int errNum;
	cl_device_id *devices;
	cl_command_queue commandQueue = NULL;
	size_t deviceBufferSize = -1;
	// 获取设备缓冲区大小
	errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
	if (deviceBufferSize <= 0)
	{
		std::cerr << "No devices available.";
		return NULL;
	}
	// 分配缓存空间
	devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
	errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
	// 选取设备中第一个
	commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);

	*device = devices[0];
	delete[] devices;
	return commandQueue;
}
// 构建程序对象
cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName)
{
	cl_int errNum;
	cl_program program;

	std::ifstream kernelFile(fileName, std::ios::in);
	if (!kernelFile.is_open())
	{
		std::cerr << "Failed to open file for reading: " << fileName << std::endl;
		return NULL;
	}
	std::ostringstream oss;
	oss << kernelFile.rdbuf();
	std::string srcStdStr = oss.str();
	const char *srcStr = srcStdStr.c_str();
	program = clCreateProgramWithSource(context, 1,
		(const char**)&srcStr,
		NULL, NULL);
	errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	return program;
}
// 构建程序对象
bool CreateMemObjects(cl_context context, cl_mem memObjects[3],
	float *a, float *b)
{
	memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * ARRAY_SIZE, a, NULL);
	memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * ARRAY_SIZE, b, NULL);
	memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,
		sizeof(float) * ARRAY_SIZE, NULL, NULL);
	return true;
}
// 释放OpenCL资源
void Cleanup(cl_context context, cl_command_queue commandQueue,
	cl_program program, cl_kernel kernel, cl_mem memObjects[3])
{
	for (int i = 0; i < 3; i++)
	{
		if (memObjects[i] != 0)
			clReleaseMemObject(memObjects[i]);
	}
	if (commandQueue != 0)
		clReleaseCommandQueue(commandQueue);

	if (kernel != 0)
		clReleaseKernel(kernel);

	if (program != 0)
		clReleaseProgram(program);

	if (context != 0)
		clReleaseContext(context);
}
int main(int argc, char** argv)
{
	cl_context context = 0;
	cl_command_queue commandQueue = 0;
	cl_program program = 0;
	cl_device_id device = 0;
	cl_kernel kernel = 0;
	cl_mem memObjects[3] = { 0, 0, 0 };
	cl_int errNum;
	// 选择OpenCL平台，创建上下文
	context = CreateContext();
	// 构建设备与命令队列
	commandQueue = CreateCommandQueue(context, &device);
	// 构建程序对象
	program = CreateProgram(context, device, "HelloWorld.cl");
	// 创建OpenCL内核，分配内存空间
	kernel = clCreateKernel(program, "hello_kernel", NULL);
	// 创建要处理的数据
	float result[ARRAY_SIZE];
	float a[ARRAY_SIZE];
	float b[ARRAY_SIZE];
	for (int i = 0; i < ARRAY_SIZE; i++)
	{
		a[i] = (float)i;
		b[i] = (float)(ARRAY_SIZE - i);
	}
	// 创建内存对象
	if (!CreateMemObjects(context, memObjects, a, b))
	{
		Cleanup(context, commandQueue, program, kernel, memObjects);
		return 1;
	}
	// 设置内核数据并执行
	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObjects[0]);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObjects[1]);
	errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjects[2]);
	size_t globalWorkSize[1] = { ARRAY_SIZE };
	size_t localWorkSize[1] = { 1 };
	errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
		globalWorkSize, localWorkSize,
		0, NULL, NULL);
	// 读取结果，释放OpenCL资源
	errNum = clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE,
		0, ARRAY_SIZE * sizeof(float), result,
		0, NULL, NULL);
	for (int i = 0; i < ARRAY_SIZE; i++)
	{
		std::cout << result[i] << " ";
	}
	std::cout << std::endl;
	std::cout << "Executed program succesfully." << std::endl;
	getchar();
	Cleanup(context, commandQueue, program, kernel, memObjects);
	return 0;
}

// 核函数文件“HelloWorld.cl”：
__kernel void hello_kernel(__global const float *a,
	__global const float *b,
	__global float *result)
{
	int gid = get_global_id(0);
	result[gid] = a[gid] + b[gid];
}
8.4　OpenCL平台编程配置与示例分析
8.4.3　平台编程示例
使用 OpenCL API C/C++，引入第三方库编程。配置 include 头文件，可在 MacOS X 10.6下OpenCL的头文件命名与其他系统不同，通常使用#if defined区分，实现代码如下所示：
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.h>
#endif
1.平台
### 查询并选择平台 
### 取得OpenCL平台
平台指OpenCL 框架, 不同的 CPU/GPU 开发商（如Intel，AMD，Nvdia），可分别定义OpenCL 框架，并查询平台。可使用 API 函数 clGetPlatformIDs 获取平台数量，实现代码如下所示: 
cl_int status = 0;
cl_uint numPlatforms;
cl_platform_id platform = NULL;
status = clGetPlatformIDs( 0, NULL, &numPlatforms);
 
if(status != CL_SUCCESS){
    printf("Error: Getting Platforms\n");
    return EXIT_FAILURE;
}
先分配内存，得到平台，而API是 clGetPlatformIDs。OpenCL先取得数目，再分配足够的内存；接着获取真正的信息。实现代码如下所示：
if (numPlatforms > 0) {
    cl_platform_id *platforms = (cl_platform_id *)malloc(numPlatforms * sizeof(cl_platform_id));
    status = clGetPlatformIDs(numPlatforms, platforms, NULL);
    if (status != CL_SUCCESS) {
        printf("Error: Getting Platform Ids.(clGetPlatformIDs)\n");
        return -1;
    }
在PC上配置Intel 的 CPU 和 AMD 的 GPU。
先用clGetPlatformInfo 得到平台信息。再通过API获取平台信息，可得到合适的平台。实现代码如下所示：
for (unsigned int i = 0; i < numPlatforms; ++i) {
        char pbuff[100];
        status = clGetPlatformInfo(
                     platforms[i],
                     CL_PLATFORM_VENDOR,
                     sizeof(pbuff),
                     pbuff,
                     NULL);
        platform = platforms[i];
        if (!strcmp(pbuff, "Advanced Micro Devices, Inc.")) {
            break;
        }
    }
应用OpenCL规格，可以获取不同的厂商信息，这里筛选AMD。同时在平台上构建上下文。其中使用了平台获取上下文属性。实现代码如下所示：
// 若找到相应平台就使用, 否则返回NULL
cl_context_properties cps[3] = {
    CL_CONTEXT_PLATFORM,
    (cl_context_properties)platform,
    0
};
cl_context_properties *cprops = (NULL == platform) ? NULL : cps;
再用 clCreateContextFromType构建上下文。实现代码如下所示：
// 生成上下文
cl_context context = clCreateContextFromType(
                         cprops,
                         CL_DEVICE_TYPE_GPU,
                         NULL,
                         NULL,
                         &status);
if (status != CL_SUCCESS) {
    printf("Error: Creating Context.(clCreateContexFromType)\n");
    return EXIT_FAILURE;
} 
其中第二个参数可以设定上下文设备类型。而本例OpenCL计算设备是GPU。其他可使用的类别如下所示： 
- CL_DEVICE_TYPE_CPU
- CL_DEVICE_TYPE_GPU
- CL_DEVICE_TYPE_ACCELERATOR
- CL_DEVICE_TYPE_DEFAULT
- CL_DEVICE_TYPE_ALL
先完成上下文构建，接着查询设备信息。实现代码如下所示：
status = clGetContextInfo(context,
                          CL_CONTEXT_DEVICES,
                          0,
                          NULL,
                          &deviceListSize);
if (status != CL_SUCCESS) {
    printf("Error: Getting Context Info device list size, clGetContextInfo)\n");
    return EXIT_FAILURE;
}
cl_device_id *devices = (cl_device_id *)malloc(deviceListSize);
if (devices == 0) {
    printf("Error: No devices found.\n");
    return EXIT_FAILURE;
}
status = clGetContextInfo(context,
                          CL_CONTEXT_DEVICES,
                          deviceListSize,
                          devices,
                          NULL);
if (status != CL_SUCCESS) {
    printf("Error: Getting Context Info (device list, clGetContextInfo)\n");
    return EXIT_FAILURE;
}
然后根据数量来分配内存，并得到所有可用的 platform平台，所使用的 API 还是clGetPlatformIDs。在 OpenCL 中，类似这样的函数调用很常见：第一次调用以取得数目，便于分配足够的内存；然后调用第二次以获取真正的信息。
调用 clGetDeviceInfo获取设备信息，包括设备类型，生产商以及设备，以及对某些扩展功能的支持与否等。 
完整实现代码如下所示：
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.h>
#endif
#include <iostream>
 
int main(int argc, char const *argv[])
{
    printf("hello OpenCL\n");
    cl_int status = 0;
    size_t deviceListSize;
    // 得到并选择可用平台
    cl_uint numPlatforms;
    cl_platform_id platform = NULL;
    status = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (status != CL_SUCCESS) {
        printf("ERROR: Getting Platforms.(clGetPlatformIDs)\n");
        return EXIT_FAILURE;
    }
    if (numPlatforms > 0) {
        cl_platform_id *platforms = (cl_platform_id *)malloc(numPlatforms * sizeof(cl_platform_id));
        status = clGetPlatformIDs(numPlatforms, platforms, NULL);
        if (status != CL_SUCCESS) {
            printf("Error: Getting Platform Ids.(clGetPlatformIDs)\n");
            return -1;
        }
		// 遍历所有平台, 选择合适的
        for (unsigned int i = 0; i < numPlatforms; ++i) {
            char pbuff[100];
            status = clGetPlatformInfo(
                         platforms[i],
                         CL_PLATFORM_VENDOR,
                         sizeof(pbuff),
                         pbuff,
                         NULL);
            platform = platforms[i];
            if (!strcmp(pbuff, "Advanced Micro Devices, Inc.")) {
                break;
            }
        }
        delete platforms;
    }
    // 找到相应平台使用, 否则返回NULL
    cl_context_properties cps[3] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platform,
        0
    };
    cl_context_properties *cprops = (NULL == platform) ? NULL : cps;
    // 生成上下文
    cl_context context = clCreateContextFromType(
                             cprops,
                             CL_DEVICE_TYPE_GPU,
                             NULL,
                             NULL,
                             &status);
    if (status != CL_SUCCESS) {
        printf("Error: Creating Context.(clCreateContexFromType)\n");
        return EXIT_FAILURE;
    }
    // 寻找OpenCL设备
    // 先得到设备列表的长度
    status = clGetContextInfo(context,
                              CL_CONTEXT_DEVICES,
                              0,
                              NULL,
                              &deviceListSize);
    if (status != CL_SUCCESS) {
        printf("Error: Getting Context Info device list size, clGetContextInfo)\n");
        return EXIT_FAILURE;
    }
    cl_device_id *devices = (cl_device_id *)malloc(deviceListSize);
    if (devices == 0) {
        printf("Error: No devices found.\n");
        return EXIT_FAILURE;
    }
    // 得到设备列表
    status = clGetContextInfo(context,
                              CL_CONTEXT_DEVICES,
                              deviceListSize,
                              devices,
                              NULL);
    if (status != CL_SUCCESS) {
        printf("Error: Getting Context Info (device list, clGetContextInfo)\n");
        return EXIT_FAILURE;
    }
2.运行时实现
实现程序实现运行时。本例是在4×4的二维空间上，同时给元素赋值。实现代码如下所示：
#define KERNEL(...)#__VA_ARGS__
const char *kernelSourceCode = KERNEL(
                                   __kernel void hellocl(__global uint *buffer)
{
    size_t gidx = get_global_id(0);
    size_t gidy = get_global_id(1);
    size_t lidx = get_local_id(0);
    buffer[gidx + 4 * gidy] = (1 << gidx) | (0x10 << gidy);
}
);
（1）加载 OpenCL 内核并创建程序对象
读取OpenCL内核，以便创建程序。实现代码如下所示：
size_t sourceSize[] = {strlen(kernelSourceCode)};
cl_program program = clCreateProgramWithSource(context,
                     1,
                     &kernelSourceCode,
                     sourceSize,
                     &status);
if (status != CL_SUCCESS) {
    printf("Error: Loading Binary into cl_program (clCreateProgramWithBinary)\n");
    return EXIT_FAILURE;
}
其中内核使用clCreateProgramWithSource。若让内核程序不公开，可生成二进制文件，接着通过 clCreateProgramWithBinary动态读入。
（2）设备编译程序中内核
内核程序读入后，再用 clBuildProgram 编译内核。实现代码如下所示：
status = clBuildProgram(program, 1, devices, NULL, NULL, NULL);
if (status != CL_SUCCESS) {
    printf("Error: Building Program (clBuildingProgram)\n");
    return EXIT_FAILURE;
}
最后，使用内核将device上对应的OpenCL，编译成机器码。
（3）创建内核对象
通过 clCreateKernel 创建内核对象。实现代码如下所示：
cl_kernel kernel = clCreateKernel(program, "hellocl", &status);
if (status != CL_SUCCESS) {
    printf("Error: Creating Kernel from program.(clCreateKernel)\n");
    return EXIT_FAILURE;
}
其中hellocl就是内核函数名。而每个内核关联程序的内核。可同时写多个内核程序，而且可以执行内核程序前建立内核对象，并多次调用 clCreateKernel 函数。
（4）创建kernel内存对象
OpenCL中的内存对象包括buffer以及image，buffer是一维数据元素的集合。image主要用来存储一维、二维、三维图像、纹理或者framebuffer。这些操作是对image对象，gpu会有优化，比如使用L1 cache，使用tile mode地址等等。实现代码如下所示：
cl_mem outputBuffer = clCreateBuffer(
									context, 
									CL_MEM_ALLOC_HOST_PTR, 
									4 * 4 * 4, 
									NULL, 
									&status);
if (status != CL_SUCCESS) {
    printf("Error: Create Buffer, outputBuffer. (clCreateBuffer)\n");
    return EXIT_FAILURE;
}
（5）设置内核参数
使用 clSetKernelArg为内核设置参数，包括设置常数，变量，内存对象。这里是内存对象，实现代码如下所示：
status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&outputBuffer);
if (status != CL_SUCCESS) {
    printf("Error: Setting kernel argument. (clSetKernelArg)\n");
    return EXIT_FAILURE;
}
每次只能设置一个参数，而多个参数需多次调用。需要设置内核参数，否则启动内核会报错。每次运行内核程序需要使用设置值，直到API重新设置参数。
（6）在device上构建命令队列
clCreateCommandQueue就是用来在某个设备上创建命令队列。设备必须为上下文关联设备，并且所有命令队列命令，都会在设备上运行。实现代码如下所示：
cl_command_queue commandQueue = clCreateCommandQueue(context,
                                devices[0],
                                0,
                                &status);
if (status != CL_SUCCESS) {
    printf("Error: Create Command Queue. (clCreateCommandQueue)\n");
    return EXIT_FAILURE;
}
（7）将内核放入命令队列
OpenCL 提供了创建内核命令的3种方案。其中最常用的内核，使用了clEnqueueNDRangeKernel 函数。实现代码如下所示：
size_t globalThreads[] = {4, 4};
size_t localThreads[] = {2, 2};
status = clEnqueueNDRangeKernel(commandQueue, kernel,
                                2, NULL, globalThreads,
                                localThreads, 0,
                                NULL, NULL);
if (status != CL_SUCCESS) {
    printf("Error: Enqueueing kernel\n");
    return EXIT_FAILURE;
}
clEnqueueNDRangeKernel 将内核传给命令队列，再调用API 并将多个内核放到命令队列 中。这里API clEnqueueTask 和 clEnqueueNativeKernel 用法类似。
调用 clFinish，确定所有队列命令执行完成。实现代码如下所示：
// 确定所有队列命令执行完成
status = clFinish(commandQueue);
if (status != CL_SUCCESS) {
    printf("Error: Finish command queue\n");
    return EXIT_FAILURE;
}
（8）将计算结果返回 host
调用 clEnqueueReadBuffer，并将 OpenCL buffer缓存计算结果返回 host。实现代码如下所示：
// 将内存对象中的结果读回Host
status = clEnqueueReadBuffer(commandQueue,
                             outputBuffer, CL_TRUE, 0,
                             4 * 4 * 4, outbuffer, 0, NULL, NULL);
if (status != CL_SUCCESS) {
    printf("Error: Read buffer queue\n");
    return EXIT_FAILURE;
}
打印运行结果：
// Host端打印结果
printf("out:\n");
for (int i = 0; i < 16; ++i) {
    printf("%x ", outbuffer[i]);
    if ((i + 1) % 4 == 0)
        printf("\n");
}
（9）资源回收
调用clRelease释放资源，完成对应C/C++的内存回收。实现代码如下所示：
// 资源回收
status = clReleaseKernel(kernel);
status = clReleaseProgram(program);
status = clReleaseMemObject(outputBuffer);
status = clReleaseCommandQueue(commandQueue);
status = clReleaseContext(context);
 
free(devices);
delete outbuffer;
（10）全部代码
本例的全部代码实现如下所示：
#include <iostream>
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.h>
#endif
#define KERNEL(...)#__VA_ARGS__
 
const char *kernelSourceCode = KERNEL(__kernel void hellocl(__global uint *buffer)
{
    size_t gidx = get_global_id(0);
    size_t gidy = get_global_id(1);
    size_t lidx = get_local_id(0);
    buffer[gidx + 4 * gidy] = (1 << gidx) | (0x10 << gidy);
}
); 
int main(int argc, char const *argv[])
{
    printf("hello OpenCL\n");
    cl_int status = 0;
    size_t deviceListSize;
 
    // 得到选择可用平台
    cl_uint numPlatforms;
    cl_platform_id platform = NULL;
    status = clGetPlatformIDs(0, NULL, &numPlatforms);
 
    if (status != CL_SUCCESS) {
        printf("ERROR: Getting Platforms.(clGetPlatformIDs)\n");
        return EXIT_FAILURE;
    }
    if (numPlatforms > 0) {
        cl_platform_id *platforms = (cl_platform_id *)malloc(numPlatforms * sizeof(cl_platform_id));
        status = clGetPlatformIDs(numPlatforms, platforms, NULL);
        if (status != CL_SUCCESS) {
            printf("Error: Getting Platform Ids.(clGetPlatformIDs)\n");
            return -1;
        }
        for (unsigned int i = 0; i < numPlatforms; ++i) {
            char pbuff[100];
            status = clGetPlatformInfo(
                         platforms[i],
                         CL_PLATFORM_VENDOR,
                         sizeof(pbuff),
                         pbuff,
                         NULL);
            platform = platforms[i];
            if (!strcmp(pbuff, "Advanced Micro Devices, Inc.")) {
                break;
            }
        }
        delete platforms;
    }
 
    // 若能得到相应平台, 直接使用, 否则返回NULL
    cl_context_properties cps[3] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platform,
        0
    };
    cl_context_properties *cprops = (NULL == platform) ? NULL : cps;
    // 生成上下文
    cl_context context = clCreateContextFromType(
                             cprops,
                             CL_DEVICE_TYPE_GPU,
                             NULL,
                             NULL,
                             &status);
    if (status != CL_SUCCESS) {
        printf("Error: Creating Context.(clCreateContexFromType)\n");
        return EXIT_FAILURE;
    }
    // 寻找OpenCL设备
    // 先计算设备列表长度
    status = clGetContextInfo(context,
                              CL_CONTEXT_DEVICES,
                              0,
                              NULL,
                              &deviceListSize);
    if (status != CL_SUCCESS) {
        printf("Error: Getting Context Info device list size, clGetContextInfo)\n");
        return EXIT_FAILURE;
    }
    cl_device_id *devices = (cl_device_id *)malloc(deviceListSize);
    if (devices == 0) {
        printf("Error: No devices found.\n");
        return EXIT_FAILURE;
    }
    // 现在得到设备列表
    status = clGetContextInfo(context,
                              CL_CONTEXT_DEVICES,
                              deviceListSize,
                              devices,
                              NULL);
    if (status != CL_SUCCESS) {
        printf("Error: Getting Context Info (device list, clGetContextInfo)\n");
        return EXIT_FAILURE;
    }
    // 装载内核程序, 编译OpenCL程序, 生成OpenCL内核实例
    size_t sourceSize[] = {strlen(kernelSourceCode)};
    cl_program program = clCreateProgramWithSource(context,
                         1,
                         &kernelSourceCode,
                         sourceSize,
                         &status);
    if (status != CL_SUCCESS) {
        printf("Error: Loading Binary into cl_program (clCreateProgramWithBinary)\n");
        return EXIT_FAILURE;
    }
    // 为指定的设备编译OpenCL程序
    status = clBuildProgram(program, 1, devices, NULL, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf("Error: Building Program (clBuildingProgram)\n");
        return EXIT_FAILURE;
    }
    // 得到指定名字的内核实例的句柄
    cl_kernel kernel = clCreateKernel(program, "hellocl", &status);
    if (status != CL_SUCCESS) {
        printf("Error: Creating Kernel from program.(clCreateKernel)\n");
        return EXIT_FAILURE;
    }
    // 创建 OpenCL buffer 对象
    unsigned int *outbuffer = new unsigned int [4 * 4];
    memset(outbuffer, 0, 4 * 4 * 4);
    cl_mem outputBuffer = clCreateBuffer(
    	context, 
    	CL_MEM_ALLOC_HOST_PTR, 
    	4 * 4 * 4, 
    	NULL, 
    	&status);
    if (status != CL_SUCCESS) {
        printf("Error: Create Buffer, outputBuffer. (clCreateBuffer)\n");
        return EXIT_FAILURE;
    }
    // 为内核程序设置参数
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&outputBuffer);
    if (status != CL_SUCCESS) {
        printf("Error: Setting kernel argument. (clSetKernelArg)\n");
        return EXIT_FAILURE;
    }
    // 创建一个OpenCL命令队列
    cl_command_queue commandQueue = clCreateCommandQueue(context,
                                    devices[0],
                                    0,
                                    &status);
    if (status != CL_SUCCESS) {
        printf("Error: Create Command Queue. (clCreateCommandQueue)\n");
        return EXIT_FAILURE;
    }
    // 将一个kernel 放入命令队列
    size_t globalThreads[] = {4, 4};
    size_t localThreads[] = {2, 2};
    status = clEnqueueNDRangeKernel(commandQueue, kernel,
                                    2, NULL, globalThreads,
                                    localThreads, 0,
                                    NULL, NULL);
    if (status != CL_SUCCESS) {
        printf("Error: Enqueueing kernel\n");
        return EXIT_FAILURE;
    }
    // 确认命令队列中所有命令都执行完毕
    status = clFinish(commandQueue);
    if (status != CL_SUCCESS) {
        printf("Error: Finish command queue\n");
        return EXIT_FAILURE;
    }
    // 将内存对象中的结果读回Host
    status = clEnqueueReadBuffer(commandQueue,
                                 outputBuffer, CL_TRUE, 0,
                                 4 * 4 * 4, outbuffer, 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf("Error: Read buffer queue\n");
        return EXIT_FAILURE;
    }
    // Host端打印结果
    printf("out:\n");
    for (int i = 0; i < 16; ++i) {
        printf("%x ", outbuffer[i]);
        if ((i + 1) % 4 == 0)
            printf("\n");
    }
    // 资源回收
    status = clReleaseKernel(kernel);
    status = clReleaseProgram(program);
    status = clReleaseMemObject(outputBuffer);
    status = clReleaseCommandQueue(commandQueue);
    status = clReleaseContext(context);
    free(devices);
    delete outbuffer;
    system("pause");
    return 0;
}
 
第9章　自动调度、自动搜索与成本模型
9.1　在CPU Auto-scheduling 自动调度
9.1.4　自动调度示例
1. 自动调度整个神经网络
包装在if __name__ == "__main__":块中。代码如下所示：
import numpy as np

import tvm
from tvm import relay, auto_scheduler
from tvm.relay import data_dep_optimization as ddo
import tvm.relay.testing
from tvm.contrib import graph_executor
需要使用Relay中继前端API定义网络。从加载一些预定义的网络tvm.relay.testing。还从MXNet，ONNX，PyTorch和TensorFlow加载模型。
对于卷积神经网络，尽管自动调度程序在任何布局下正常工作，使用NHWC布局通常实现最佳性能。使用自动调度程序对NHWC布局实施了更多优化。建议将模型转换为NHWC布局以使用自动调度程序。使用ConvertLayout传递在TVM中进行布局转换。包括以下功能模块： 
1）	获取网络的符号定义和随机权重。
2）	定义神经网络和编译目标。
实现代码如下所示：
def get_network(name, batch_size, layout="NHWC", dtype="float32", use_sparse=False):
    // 获取网络的符号定义和随机权重
    # auto-scheduler首选NHWC布局
    if layout == "NHWC":
        image_shape=(224, 224, 3)
    elif layout == "NCHW":
        image_shape=(3, 224, 224)
    else:
        raise ValueError("Invalid layout: "+layout)

    input_shape=(batch_size, )+image_shape
    output_shape=(batch_size, 1000)

    if name.startswith("resnet-"):
        n_layer=int(name.split("-")[1])
        mod, params=relay.testing.resnet.get_workload(
            num_layers=n_layer, 
            batch_size=batch_size, 
            layout=layout, 
            dtype=dtype, 
            image_shape=image_shape, 
      )
    elif name.startswith("resnet3d-"):
        n_layer=int(name.split("-")[1])
        mod, params=relay.testing.resnet.get_workload(
            num_layers=n_layer, 
            batch_size=batch_size, 
            layout=layout, 
            dtype=dtype, 
            image_shape=image_shape, 
      )
    elif name == "mobilenet":
        mod, params=relay.testing.mobilenet.get_workload(
            batch_size=batch_size, layout=layout, dtype=dtype, image_shape=image_shape
      )
    elif name == "squeezenet_v1.1":
        assert layout == "NCHW", "squeezenet_v1.1 only supports NCHW layout"
        mod, params=relay.testing.squeezenet.get_workload(
            version="1.1", 
            batch_size=batch_size, 
            dtype=dtype, 
            image_shape=image_shape, 
      )
    elif name == "inception_v3":
        input_shape=(batch_size, 3, 299, 299) if layout == "NCHW" else (batch_size, 299, 299, 3)
        mod, params=relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == "mxnet":
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model
        assert layout == "NCHW"
        block=get_model("resnet50_v1", pretrained=True)
        mod, params=relay.frontend.from_mxnet(block, shape={"data": input_shape}, dtype=dtype)
        net=mod["main"]
        net=relay.Function(
            net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
      )
        mod=tvm.IRModule.from_expr(net)
    elif name == "mlp":
        mod, params=relay.testing.mlp.get_workload(
            batch_size=batch_size, dtype=dtype, image_shape=image_shape, num_classes=1000
      )
    else:
        raise ValueError("Network not found.")
    if use_sparse:
        from tvm.topi.sparse.utils import convert_model_dense_to_sparse
        mod, params=convert_model_dense_to_sparse(mod, params, bs_r=4, random_params=True)
    return mod, params, input_shape, output_shape

# 定义神经网络和编译目标
# 如果目标机器支持avx512指令，可用"llvm -mcpu=skylake-avx512"替换"llvm -mcpu=core-avx2"
network="resnet-50"
use_sparse=False
batch_size=1
layout="NHWC"
target=tvm.target.Target("llvm -mcpu=core-avx2")
dtype="float32"
log_file="%s-%s-B%d-%s.json" % (network, layout, batch_size, target.kind.name)
2. 提取搜索任务
可从网络中提取搜索任务及权重，再将网络延迟近似为sum(latency[t] * weight[t])，其中latency[t]是任务的延迟，weight[t]是任务的权重，这是任务调度器优化目标。实现代码如下所示：
# 从网络中提取任务
print("Get model...")
mod, params, input_shape, output_shape=get_network(
    network, 
    batch_size, 
    layout, 
    dtype=dtype, 
    use_sparse=use_sparse, 
)
print("Extract tasks...")
tasks, task_weights=auto_scheduler.extract_tasks(mod["main"], params, target)
for idx, task in enumerate(tasks):
    print("Task %d  (workload key: %s) " % (idx, task.workload_key))
    print(task.compute_dag)
3. 开始启动调度
    实现代码如下所示：
def run_tuning():
    print("Begin tuning...")
    tuner=auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option=auto_scheduler.TuningOptions(
        num_measure_trials=200,  # change this to 20000 to achieve the best performance
        runner=auto_scheduler.LocalRunner(repeat=10, enable_cpu_cache_flush=True), 
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)], 
  )

    if use_sparse:
        from tvm.topi.sparse.utils import sparse_sketch_rules

        search_policy=[
            auto_scheduler.SketchPolicy(
                task, 
                program_cost_model=auto_scheduler.XGBModel(), 
                init_search_callbacks=sparse_sketch_rules(), 
          )
            for task in tasks
       ]

        tuner.tune(tune_option, search_policy=search_policy)
    else:
        tuner.tune(tune_option)

# 不在网页服务器中运行调优，因为这需要很长时间
# 注销注释以下行，以便自行运行
# run_tuning()
打印调试跟踪信息。在调优过程中，会打印很多信息。最重要的信息是任务调度程序的输出。
4. 编译与评估
实现代码如下所示：
# 以历史最好的方式编译
print("Compile...")
with auto_scheduler.ApplyHistoryBest(log_file):
    with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
        lib=relay.build(mod, target=target, params=params)

# 创建图形执行器
dev=tvm.device(str(target), 0)
module=graph_executor.GraphModule(lib["default"](dev))
data_tvm=tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
module.set_input("data", data_tvm)

# 预估
print("Evaluate inference time cost...")
print(module.benchmark(dev, repeat=3, min_repeat_ms=500))
9.2　AutoTVM用成本模型自动搜索
9.2.3　AutoTVM自动搜索示例
1. tvm 使用 autotvm 包，需要安装依赖项
pip3 install --user psutil xgboost tornado cloudpickle
如果tvm的FFI用cython，那么TVM调优速度会更快，可在tvm根目录执行。
pip3 install --user cython
sudo make cython3
2. 回到Python代码，导入包
    实现代码如下所示：
import logging
import sys
import numpy as np
import tvm
from tvm import te, topi, testing
from tvm.topi.testing import conv2dConv2d_nchw_python
import tvm.testing
from tvm import autotvm
9.2.4　定义搜索空间
首先，使用Relay前端导入现有模型。在这里，以带有(1, 3, 224, 224)输入形状的MXNet模型为例。实现代码如下所示：
sym, arg_params, aux_params = mxnet.model.load_checkpoint(model_path, epoch)
net, params = relay.from_mxnet(sym, shape={'data': (1, 3, 224, 224)}, arg_params=arg_params, aux_params=aux_params)
接下来，可使用Relay量化 API，并且转换为量化模型。实现代码如下所示：
net = relay.quantize.quantize(net, params=params)
然后，使用 AutoTVM 为模型中的算子提取调优任务并进行自动优化。这里AutoTVM为此提供了一个示例。
最后，建立模型并在量化模式下运行推理。实现代码如下所示：
with relay.build_config(opt_level=3):
    graph, lib, params = relay.build(net, target)
Conv2d 有很大算子搜索空间，如某些shape，精度范围在 10^9 的级别。实现代码如下所示：
@autotvm.template("tutorial/conv2dConv2d_no_batching")
def conv2dConv2d_no_batching(N, H, W, CO, CI, KH, KW, stride, padding):
    assert N == 1, "Only consider batch_size=1 in this template"

    data=te.placeholder((N, CI, H, W), name="data")
    kernel=te.placeholder((CO, CI, KH, KW), name="kernel")
    conv=topi.nn.conv2dConv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    s=te.create_schedule([conv.op])

    ##### 空间定义开始 #####
    n, f, y, x=s[conv].op.axis
    rc, ry, rx=s[conv].op.reduce_axis

    cfg=autotvm.get_config()
    cfg.define_split("tile_f", f, num_outputs=4)
    cfg.define_split("tile_y", y, num_outputs=4)
    cfg.define_split("tile_x", x, num_outputs=4)
    cfg.define_split("tile_rc", rc, num_outputs=3)
    cfg.define_split("tile_ry", ry, num_outputs=3)
    cfg.define_split("tile_rx", rx, num_outputs=3)
    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])
    cfg.define_knob("unroll_explicit", [0, 1])
    ##### 空间定义结束 #####

    # 内联填充
    pad_data=s[conv].op.input_tensors[0]
    s[pad_data].compute_inline()
    data, raw_data=pad_data, data

    output=conv
    OL=s.cache_write(conv, "local")

    # 创建缓存阶段
    AA=s.cache_read(data, "shared", [OL])
    WW=s.cache_read(kernel, "shared", [OL])
    AL=s.cache_read(AA, "local", [OL])
    WL=s.cache_read(WW, "local", [OL])

    # 平铺和绑定空间轴
    n, f, y, x=s[output].op.axis
    bf, vf, tf, fi=cfg["tile_f"].apply(s, output, f)
    by, vy, ty, yi=cfg["tile_y"].apply(s, output, y)
    bx, vx, tx, xi=cfg["tile_x"].apply(s, output, x)
    kernel_scope=n  # this is the scope to attach global config inside this kernel

    s[output].bind(bf, te.thread_axis("blockIdx.z"))
    s[output].bind(by, te.thread_axis("blockIdx.y"))
    s[output].bind(bx, te.thread_axis("blockIdx.x"))
    s[output].bind(vf, te.thread_axis("vthread"))
    s[output].bind(vy, te.thread_axis("vthread"))
    s[output].bind(vx, te.thread_axis("vthread"))
    s[output].bind(tf, te.thread_axis("threadIdx.z"))
    s[output].bind(ty, te.thread_axis("threadIdx.y"))
    s[output].bind(tx, te.thread_axis("threadIdx.x"))
    s[output].reorder(n, bf, by, bx, vf, vy, vx, tf, ty, tx, fi, yi, xi)
    s[OL].compute_at(s[output], tx)

    # 平铺规约轴
    n, f, y, x=s[OL].op.axis
    rc, ry, rx=s[OL].op.reduce_axis
    rco, rcm, rci=cfg["tile_rc"].apply(s, OL, rc)
    ryo, rym, ryi=cfg["tile_rx"].apply(s, OL, ry)
    rxo, rxm, rxi=cfg["tile_ry"].apply(s, OL, rx)
    s[OL].reorder(rco, ryo, rxo, rcm, rym, rxm, rci, ryi, rxi, n, f, y, x)

    s[AA].compute_at(s[OL], rxo)
    s[WW].compute_at(s[OL], rxo)
    s[AL].compute_at(s[OL], rxm)
    s[WL].compute_at(s[OL], rxm)

    # 协同抓取
    for load in [AA, WW]:
        n, f, y, x=s[load].op.axis
        fused=s[load].fuse(n, f, y, x)
        tz, fused=s[load].split(fused, nparts=cfg["tile_f"].size[2])
        ty, fused=s[load].split(fused, nparts=cfg["tile_y"].size[2])
        tx, fused=s[load].split(fused, nparts=cfg["tile_x"].size[2])
        s[load].bind(tz, te.thread_axis("threadIdx.z"))
        s[load].bind(ty, te.thread_axis("threadIdx.y"))
        s[load].bind(tx, te.thread_axis("threadIdx.x"))

    # 调整展开
    s[output].pragma(kernel_scope, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
    s[output].pragma(kernel_scope, "unroll_explicit", cfg["unroll_explicit"].val)

return s, [raw_data, kernel, conv]
9.2.5　用成本模型进行搜索处理
实现代码如下所示：
# logging config (for printing tuning log to screen)
logging.getLogger("autotvm").setLevel(logging.DEBUG)
logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

# resnet的最后一层
N, H, W, CO, CI, KH, KW, strides, padding=1, 7, 7, 512, 512, 3, 3, (1, 1), (1, 1)
task=autotvm.task.create(
    "tutorial/conv2dConv2d_no_batching", args=(N, H, W, CO, CI, KH, KW, strides, padding), target="cuda"
)
print(task.config_space)

# 使用本地GPU，每次配置测量10次，以减少差异
# 编译程序的超时为10秒，而运行超时为4秒
measure_option=autotvm.measure_option(
    builder=autotvm.LocalBuilder(), 
    runner=autotvm.LocalRunner(repeat=3, min_repeat_ms=100, timeout=4), 
)

# 开始调整，将log记录到文件“conv2dConv2d.log”
# 在调优期间，将尝试许多无效的配置，因此将看到许多错误报告。一旦能看到非零
# GFLOP，就是好的。 
tuner=autotvm.tuner.XGBTuner(task)
tuner.tune(
    n_trial=20, 
    measure_option=measure_option, 
    callbacks=[autotvm.callback.log_to_file("conv2dConv2d.log")], 
)
从日志文件测试验证。
# 检查最佳配置
dispatch_context=autotvm.apply_history_best("conv2dConv2d.log")
best_config=dispatch_context.query(task.target, task.workload)
print("\nBest config:")
print(best_config)

# 从日志文件应用最佳历史记录
with autotvm.apply_history_best("conv2dConv2d.log"):
    with tvm.target.Target("cuda"):
        s, arg_bufs=conv2dConv2d_no_batching(N, H, W, CO, CI, KH, KW, strides, padding)
        func=tvm.build(s, arg_bufs)

# 检查正确性
a_np=np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
w_np=np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
c_np=conv2dConv2d_nchw_python(a_np, w_np, strides, padding)

dev=tvm.cuda()
a_tvm=tvm.nd.array(a_np, device=dev)
w_tvm=tvm.nd.array(w_np, device=dev)
c_tvm=tvm.nd.empty(c_np.shape, device=dev)
func(a_tvm, w_tvm, c_tvm)

tvm.testing.assert_allclose(c_np, c_tvm.numpy(), rtol=1e-2)

# 评估运行时间。选择了一个大的重复数（400），以减少噪声和内核启动的开销。
# 可以使用nvprof验证结果
evaluator=func.time_evaluator(func.entry_name, dev, number=400)
print("Time cost of this operator: %f" % evaluator(a_tvm, w_tvm, c_tvm).mean)
输出结果：
Best config:
[('tile_f', [-1, 1, 4, 1]), ('tile_y', [-1, 1, 1, 1]), ('tile_x', [-1, 7, 1, 1]), ('tile_rc', [-1, 4, 1]), ('tile_ry', [-1, 1, 1]), ('tile_rx', [-1, 1, 3]), ('auto_unroll_max_step', 1500), ('unroll_explicit', 1)]
 

