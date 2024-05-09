#include <iostream>
#include <chrono>
#include <random>
#include "ddimscheduler.hpp"
using namespace Scheduler;

int main() {

    auto scheduler = DDIMScheduler("scheduler_config.json");
    scheduler.set_timesteps(10);
    std::vector<int> timesteps;
    scheduler.get_timesteps(timesteps);

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::normal_distribution<float> distribution(.0, 1.0);

    std::vector<float> sample(1 * 4 * 64 * 64);
    std::vector<float> model_output(1 * 4 * 64 * 64);

    for(int i = 0; i < 4 * 64 * 64; i++){
        sample[i] = distribution(generator);
        model_output[i] = distribution(generator);
    }

    std::vector<float> pred_sample;

    for(auto t: timesteps){
        scheduler.step(model_output, {1, 4, 3, 3}, sample, {1, 4, 3, 3}, pred_sample, t);
    }
    std::cout << "passed!" << std::endl;
    return 0;
}
