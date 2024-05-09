## ddim_scheduler_cpp


This project provides a cross-platform C++ ddim scheduler library that can be universally deployed. It is consistent with the [Diffusers:DDIMScheduler](https://huggingface.co/docs/diffusers/api/schedulers/ddim) interface. You can easily convert Python code to C++.

ddim_scheduler_cpp是一个C++版本的ddim-scheduler库。矩阵运算使用了Eigen库，所以理论上是支持各个平台的。ddim_scheduler_cpp提供了与[Diffusers:DDIMScheduler](https://huggingface.co/docs/diffusers/api/schedulers/ddim) 相同的接口，可以直接拿来替换python版本。


### Getting Started

#### build

`mkdir build & cd build`

`cmake .. -DDDIM_SHARED_LIB=ON/OFF -DCMAKE_INSTALL_PREFIX="path you wanna install"`

`make -j8`


After install

```asm
install/
├── ddim_scheduler_cpp
    └── ddimscheduler.hpp
└── lib   
    └── libddim_scheduler_cpp.a or libddim_scheduler_cpp.so
```



#### Example Code

You can get [scheduler_config.json](https://huggingface.co/runwayml/stable-diffusion-inpainting/blob/main/scheduler/scheduler_config.json) from huggingface. 

```c++
// init from json
auto scheduler = DDIMScheduler("scheduler_config.json");

// set num_inference_steps
scheduler.set_timesteps(10);

// get timesteps
std::vector<int> timesteps;
scheduler.get_timesteps(timesteps);

// random init for example
std::vector<float> sample(1 * 4 * 64 * 64);
std::vector<float> model_output(1 * 4 * 64 * 64);

for(int i = 0; i < 4 * 64 * 64; i++){
    sample[i] = distribution(generator);
    model_output[i] = distribution(generator);
}

// step
std::vector<float> pred_sample;
for(auto t: timesteps){
    scheduler.step(model_output, {1, 4, 3, 3}, sample, {1, 4, 3, 3}, pred_sample, t);
}
```

