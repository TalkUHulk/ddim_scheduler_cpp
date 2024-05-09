//
// Created by TalkUHulk on 2024/5/5.
//

#include "schedulerconfig.hpp"
#include "nlohmann/json.hpp"
#include <fstream>

using json = nlohmann::json;

namespace Scheduler {

    DDIMMeta::DDIMMeta(const std::string &config) {
        std::ifstream f(config);
        json json_config = json::parse(f);
        f.close();
        num_train_timesteps = json_config["num_train_timesteps"].get<int>();

        beta_schedule = json_config["beta_schedule"].get<std::string>();
        auto beta_start = json_config["beta_start"].get<float>();
        auto beta_end = json_config["beta_end"].get<float>();
        if (beta_schedule == "linear")
            betas = Eigen::VectorXf::LinSpaced(num_train_timesteps, beta_start, beta_end);
        else if (beta_schedule == "scaled_linear") {
            betas = Eigen::VectorXf::LinSpaced(num_train_timesteps, sqrt(beta_start), sqrt(beta_end));
            betas = betas.array().pow(2);
        }
        alphas = 1.0f - betas.array();

        alphas_cumprod.resize(alphas.size());
        alphas_cumprod[0] = alphas[0];
        for (int i = 1; i < alphas.size(); i++) {
            alphas_cumprod[i] = alphas_cumprod[i - 1] * alphas[i];
        }

        // For the final step, there is no previous alphas_cumprod because we are already at 0
        // set_alpha_to_one` decides whether we set this parameter simply to one or
        // whether we use the final alpha of the "non-previous" one.
        auto set_alpha_to_one = json_config["set_alpha_to_one"].get<bool>();
        final_alpha_cumprod = set_alpha_to_one ? 1.f : float(alphas_cumprod[0]);

        timesteps = Eigen::VectorXi::LinSpaced(num_train_timesteps, 0, num_train_timesteps - 1).reverse();
//
        steps_offset = json_config["steps_offset"].get<int>();
        timestep_spacing = json_config["timestep_spacing"].get<std::string>();
        prediction_type = json_config["prediction_type"].get<std::string>();

        clip_sample = json_config["clip_sample"].get<bool>();
        clip_sample_range = json_config["clip_sample_range"].get<float>();
        thresholding = json_config["thresholding"].get<bool>();
        dynamic_thresholding_ratio = json_config["dynamic_thresholding_ratio"].get<float>();
        sample_max_value = json_config["sample_max_value"].get<float>();

    }
}