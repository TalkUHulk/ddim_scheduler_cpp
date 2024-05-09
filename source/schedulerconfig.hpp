//
// Created by TalkUHulk on 2024/5/5.
//

#ifndef DDIM_SCHEDULER_CPP_SCHEDULERCONFIG_HPP
#define DDIM_SCHEDULER_CPP_SCHEDULERCONFIG_HPP

#include <Eigen/Eigen>
#include <string>

namespace Scheduler {

    struct DDIMMeta {
        Eigen::VectorXf betas;
        Eigen::VectorXf alphas;
        Eigen::VectorXf alphas_cumprod; // cumulative product of alphas
        Eigen::VectorXi timesteps;

        int num_train_timesteps = 1000;
        int steps_offset = 1;
        std::string timestep_spacing = "leading";
        std::string prediction_type = "epsilon";
        std::string beta_schedule = "linear";
        bool clip_sample = false;
        float clip_sample_range = 1.0f;
        bool thresholding = false;
        float dynamic_thresholding_ratio = 0.995;
        float sample_max_value = 1.0f;
        float final_alpha_cumprod = .0;
        float init_noise_sigma = 1.0f;

        explicit DDIMMeta(const std::string &config);
        ~DDIMMeta() = default;
    };

}
#endif //DDIM_SCHEDULER_CPP_SCHEDULERCONFIG_HPP
