//
// Created by TalkUHulk on 2024/4/25.
//

#include "ddimscheduler.hpp"
#include "schedulerconfig.hpp"
#include <unsupported/Eigen/CXX11/Tensor>
#include <cmath>


namespace Scheduler {

#define LOG_PRINT(format, ...) printf(format, ##__VA_ARGS__)
#define LOG_ERROR(format, ...) printf(format, ##__VA_ARGS__)

    float DDIMScheduler::get_variance(int timestep, int prev_timestep) {
        auto alpha_prod_t = meta_ptr->alphas_cumprod[timestep];
        auto alpha_prod_t_prev = prev_timestep >= 0 ? meta_ptr->alphas_cumprod[prev_timestep] : meta_ptr->final_alpha_cumprod;
        auto beta_prod_t = 1 - alpha_prod_t;
        auto beta_prod_t_prev = 1 - alpha_prod_t_prev;

        auto variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev);
        return variance;
    }

    DDIMScheduler::DDIMScheduler(const std::string &config) {
        meta_ptr = new DDIMMeta(config);
    }

/*
 * num_inference_steps: The number of diffusion steps used when generating samples with a pre-trained model.
 * */
    int DDIMScheduler::set_timesteps(int num_inference_steps) {
        if (num_inference_steps > meta_ptr->num_train_timesteps) {
            LOG_ERROR("num_inference_steps:%d cannot be larger than num_train_timesteps:%d!\n", num_inference_steps, meta_ptr->num_train_timesteps);
            return -1;
        }
        this->num_inference_steps = num_inference_steps;

        if (meta_ptr->timestep_spacing == "linspace") {
            meta_ptr->timesteps = Eigen::VectorXi::LinSpaced(num_inference_steps, 0, meta_ptr->num_train_timesteps - 1).reverse();
        } else if (meta_ptr->timestep_spacing == "leading") {
            int step_ratio = floor(meta_ptr->num_train_timesteps / num_inference_steps);
            meta_ptr->timesteps = Eigen::VectorXi::LinSpaced(num_inference_steps, 0, num_inference_steps - 1).reverse().array();
            meta_ptr->timesteps = meta_ptr->timesteps.array() * step_ratio + meta_ptr->steps_offset;
        } else if (meta_ptr->timestep_spacing == "trailing") {
            auto step_ratio = float(meta_ptr->num_train_timesteps) / num_inference_steps;
            meta_ptr->timesteps = Eigen::VectorXi::LinSpaced(num_inference_steps, -meta_ptr->num_train_timesteps, -step_ratio);
            meta_ptr->timesteps = -meta_ptr->timesteps.array() - 1;
        } else {
            LOG_ERROR("%s is not supported. Please make sure to choose one of 'leading' or 'trailing'.\n", meta_ptr->timestep_spacing.c_str());
            return -1;
        }
        return 0;
    }

    void DDIMScheduler::get_timesteps(std::vector<int> &dst) {
        assert(meta_ptr);
        dst.assign(meta_ptr->timesteps.begin(), meta_ptr->timesteps.end());
    }

    float DDIMScheduler::get_init_noise_sigma() const {
        assert(meta_ptr);
        return meta_ptr->init_noise_sigma;
    }

    int DDIMScheduler::step(std::vector<float> &model_output, const std::vector<int> &model_output_size,
                            std::vector<float> &sample, const std::vector<int> &sample_size,
                            std::vector<float> &prev_sample,
                            int timestep, float eta, bool use_clipped_model_output) {
        if (num_inference_steps == 0) {
            LOG_ERROR("Number of inference steps is 0, you need to run 'set_timesteps' after creating the scheduler.\n");
            return -1;
        }
        assert(meta_ptr);
        //step 1. get previous step value (=t-1)
        int prev_timestep = timestep - meta_ptr->num_train_timesteps / num_inference_steps;

        //step 2. compute alpha_cumprod_t & alpha_cumprod_t_prev

        auto alpha_prod_t = (float) meta_ptr->alphas_cumprod[timestep];
        auto alpha_prod_t_prev = float(prev_timestep >= 0 ? meta_ptr->alphas_cumprod[prev_timestep] : meta_ptr->final_alpha_cumprod);


        auto beta_prod_t = 1 - alpha_prod_t;

        // step 3. compute predicted original sample from predicted noise also called
        // "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf

        Eigen::TensorMap<Eigen::Tensor<float, 4, Eigen::RowMajor>> model_output_t(model_output.data(),
                                                                                  model_output_size[0],
                                                                                  model_output_size[1],
                                                                                  model_output_size[2],
                                                                                  model_output_size[3]);

        Eigen::TensorMap<Eigen::Tensor<float, 4, Eigen::RowMajor>> sample_t(sample.data(),
                                                                            sample_size[0], sample_size[1],
                                                                            sample_size[2], sample_size[3]);

        // 与numpy对比方便，采用行优先
        Eigen::Tensor<float, 4, Eigen::RowMajor> pred_original_sample, pred_epsilon; // epsilon:正态分布噪声

        if (meta_ptr->prediction_type == "epsilon") {
            pred_original_sample = (sample_t - sqrt(beta_prod_t) * model_output_t) / sqrt(alpha_prod_t);
            pred_epsilon = model_output_t;
        } else if (meta_ptr->prediction_type == "sample") {
            pred_original_sample = model_output_t;
            pred_epsilon = (sample_t - sqrt(alpha_prod_t) * pred_original_sample) / sqrt(beta_prod_t);
        } else if (meta_ptr->prediction_type == "v_prediction") {
            pred_original_sample = sqrt(alpha_prod_t) * sample_t - sqrt(beta_prod_t) * model_output_t;
            pred_epsilon = sqrt(alpha_prod_t) * model_output_t + sqrt(beta_prod_t) * sample_t;
        } else {
            LOG_ERROR("prediction_type: %s must be one of `epsilon`, `sample`, or `v_prediction`.\n", meta_ptr->prediction_type.c_str());
            return -1;
        }

        // step 4. Clip or threshold "predicted x_0"
        if (meta_ptr->thresholding) {
            // only support batch-size 1
            /*
             "Dynamic thresholding: At each sampling step we set s to a certain percentile absolute pixel value in xt0 (the
            prediction of x_0 at timestep t), and if s > 1, then we threshold xt0 to the range [-s, s] and then divide by
            s. Dynamic thresholding pushes saturated pixels (those near -1 and 1) inwards, thereby actively preventing
            pixels from saturation at each step. We find that dynamic thresholding results in significantly better
            photorealism as well as better image-text alignment, especially when using very large guidance weights."

            https://arxiv.org/abs/2205.11487
             */
            Eigen::Map<Eigen::VectorXf> pred_original_sample_flatten(pred_original_sample.data(),
                                                                     pred_original_sample.size());
            // pred_original_sample_flatten will change pred_original_sample, so copy.
            Eigen::VectorXf pred_original_sample_flatten_sort = pred_original_sample_flatten.cwiseAbs();
            std::sort(pred_original_sample_flatten_sort.begin(), pred_original_sample_flatten_sort.end(),
                      std::less<float>());

            auto q = meta_ptr->dynamic_thresholding_ratio * (pred_original_sample_flatten_sort.size() - 1);
            int index = floor(q);
            auto quantile = pred_original_sample_flatten_sort[index]
                            +
                            (pred_original_sample_flatten_sort[index + 1] - pred_original_sample_flatten_sort[index]) *
                            (q - index);
            quantile = fmin(1, fmax(quantile, meta_ptr->sample_max_value));
            pred_original_sample = pred_original_sample.clip(-quantile, quantile);

        } else if (meta_ptr->clip_sample) {
            pred_original_sample = pred_original_sample.clip(-meta_ptr->clip_sample_range, meta_ptr->clip_sample_range);
        }

        // step 5. compute variance: "sigma_t(η)" -> see formula (16)
        // σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        auto variance = get_variance(timestep, prev_timestep);
        auto std_dev_t = eta * sqrt(variance); //eta 默认为0。全程推导并未使用，std_dev_t取值可以为0。

        if (use_clipped_model_output) {
            // the pred_epsilon is always re-derived from the clipped x_0 in Glide
            pred_epsilon = (sample_t - sqrt(alpha_prod_t) * pred_original_sample) / sqrt(beta_prod_t);
        }

        // step 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        auto pred_sample_direction = sqrt((1 - alpha_prod_t_prev - sqrt(std_dev_t))) * pred_epsilon;

        // step 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        Eigen::Tensor<float, 4, Eigen::RowMajor> prev_sample_t =
                sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction;
        prev_sample.clear();
        prev_sample.resize(prev_sample_t.size());
        prev_sample.assign(prev_sample_t.data(), prev_sample_t.data() + prev_sample_t.size());
        return 0;
    }

    DDIMScheduler::~DDIMScheduler() {
        delete meta_ptr;
        meta_ptr = nullptr;
    }

    int DDIMScheduler::add_noise(std::vector<float> &sample, const std::vector<int> &sample_size,
                                 std::vector<float> &noise, const std::vector<int> &noise_size, int timesteps,
                                 std::vector<float> &noisy_samples) {
        assert(meta_ptr);
        if(sample_size.size() != noise_size.size()){
            LOG_ERROR("Sample and noise must has the same shape.\n");
            return -1;
        }
        for(int i = 0; i < sample_size.size(); i++){
            if(sample_size[i] != noise_size[i]){
                LOG_ERROR("Sample and noise must has the same shape.\n");
                return -1;
            }
        }

        auto sqrt_alpha_prod = sqrt(meta_ptr->alphas_cumprod[timesteps]);
        auto sqrt_one_minus_alpha_prod = sqrt(1 - meta_ptr->alphas_cumprod[timesteps]);

        auto length = std::accumulate(sample_size.begin(), sample_size.end(), 1, std::multiplies<int>());

        noisy_samples.clear();
        noisy_samples.resize(length);

        Eigen::TensorMap<Eigen::Tensor<float, 4, Eigen::RowMajor>> mat_sample(sample.data(),
                                                                              sample_size[0], sample_size[1],
                                                                              sample_size[2], sample_size[3]);
        Eigen::TensorMap<Eigen::Tensor<float, 4, Eigen::RowMajor>> mat_noise(noise.data(),
                                                                              sample_size[0], sample_size[1],
                                                                              sample_size[2], sample_size[3]);
        Eigen::Tensor<float, 4, Eigen::RowMajor> result = mat_sample * sqrt_alpha_prod + mat_noise * sqrt_one_minus_alpha_prod;
        noisy_samples.assign(result.data(), result.data() + result.size());
//        for(int i = 0; i < length; i++){
//            noisy_samples[i] = sqrt_alpha_prod * sample[i] + sqrt_one_minus_alpha_prod * noise[i];
//        }
        return 0;
    }


}


