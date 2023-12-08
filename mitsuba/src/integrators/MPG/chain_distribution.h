#pragma once

#include "util.h"

NAMESPACE_BEGIN(mitsuba)
template <typename Float, typename Spectrum> class ChainDistribution;

template <typename Float, typename Spectrum>
std::shared_ptr<ChainDistribution<Float, Spectrum>> chainDistributionFactory();

template <typename Float, typename Spectrum> struct ChainDistributionSamplingContext {
    MTS_IMPORT_TYPES();

    // for bounce
    DiscreteDistribution<Float> bounce_distribution;

    // for type
    DiscreteDistribution<Float> tau_distribution_per_bounce[MAX_CHAIN_LENGTH];
    std::vector<int> value_tau[MAX_CHAIN_LENGTH];

    // for direction
    std::map<int, DiscreteDistribution<Float>> direction_distribution_for_type;

    SurfaceInteraction3f si;

    void clear() {
        direction_distribution_for_type.clear();
        for (int i = 0; i < MAX_CHAIN_LENGTH; i++)
            value_tau[i].clear();
    }
};

template <typename Float, typename Spectrum> class ChainDistribution {
public:
    MTS_IMPORT_TYPES();
    using SubpathSample                    = SubpathSample<Float, Spectrum>;
    using Sampler                          = Sampler<Float, Spectrum>;
    using DTreeWrapper                     = DTreeWrapper<Float, Spectrum>;
    using DTreeRecord                      = DTreeRecord<Float, Spectrum>;
    using ChainDistributionSamplingContext = ChainDistributionSamplingContext<Float, Spectrum>;
    using AABB                             = BoundingBox<Point6f>;

    ChainDistribution() {}

    ChainDistribution(const ChainDistribution &other) : dtree(other.dtree), dtree_for_type(other.dtree_for_type) {}

    Point6f point3f_to_6f(Point3f p) { return { p[0], p[1], p[2], p[0], p[1], p[2] }; }

    Point6f point3f_to_6f(Point3f p, Point3f q) { return { p[0], p[1], p[2], q[0], q[1], q[2] }; }

    void record_no_lock(SubpathSample rec, Float statisticalWeight) {
        if (rec.bounce >= MAX_CHAIN_LENGTH) {
            return;
        }
        if (std::isinf(rec.energy) || std::isnan(rec.energy)) {
            return;
        }
        rec.energy *= statisticalWeight;
        weight += statisticalWeight;
        if (!rec.filtered) {
            weight_not_filtered += statisticalWeight;
        }
        aabb.expand(point3f_to_6f(rec.xD, rec.xL));
        if (global_sms_config.ablation == 2) {
            rec.type = -1;
        }
        m_subpath_sample_list.push_back(rec);
    }

    void record(SubpathSample rec, Float statisticalWeight) {
        mu.lock();
        record_no_lock(rec, statisticalWeight);
        mu.unlock();
    }

    void load_samples(const std::vector<SubpathSample> &subpath_sample_list) {
        m_subpath_sample_list.clear();
        weight              = 0;
        weight_not_filtered = 0;
        for (auto i : subpath_sample_list) {
            record_no_lock(i, 1.f);
        }
    }

    void build_distribution() {
        if (m_subpath_sample_list.size() == 0) {
            return;
        }

        NanosecondTimer timer;

        for (int i = 0; i < MAX_CHAIN_LENGTH; i++) {
            value_tau[i].clear();
        }
        direction_sample_for_type.clear();
        cached_radius_for_type.clear();
        cached_radius_for_type_mu.clear();

        std::vector<Float> pmf_n(MAX_CHAIN_LENGTH);
        std::vector<std::vector<Float>> pmf_tau_per_bounce(MAX_CHAIN_LENGTH);
        std::map<int, std::vector<Float>> pmf_sample_for_type;
        total_energy = 0;

        for (SubpathSample &subpath_sample : m_subpath_sample_list) {
            if (subpath_sample.bounce >= MAX_CHAIN_LENGTH) {
                continue;
            }
            if (std::isinf(subpath_sample.energy) || std::isnan(subpath_sample.energy)) {
                continue;
            }

            // for bounce sampling
            pmf_n[subpath_sample.bounce] += subpath_sample.energy;
            total_energy += subpath_sample.energy;

            // for tau sampling (given a specific bounce)
            pmf_tau_per_bounce[subpath_sample.bounce].push_back(subpath_sample.energy);
            value_tau[subpath_sample.bounce].push_back(subpath_sample.type);

            // for direction sampling (given a specific tau)
            if (global_sms_config.directional_struct == 0) { // ours
                pmf_sample_for_type[subpath_sample.type].push_back(subpath_sample.energy);
                cached_radius_for_type[subpath_sample.type].push_back(-1);
                if (cached_radius_for_type_mu[subpath_sample.type] == nullptr)
                    cached_radius_for_type_mu[subpath_sample.type] = std::make_shared<std::mutex>();
                direction_sample_for_type[subpath_sample.type].push_back(
                    normalize(subpath_sample.x1 - subpath_sample.xD));
            } else if (global_sms_config.directional_struct == 1) { // dtree
                DTreeRecord rec = { normalize(subpath_sample.x1 - subpath_sample.xD),
                                    subpath_sample.energy,
                                    subpath_sample.energy,
                                    1.0f,
                                    1.0f,
                                    1.0f,
                                    1.0f,
                                    false,
                                    subpath_sample.type };
                drecs.push_back(rec);
            }
        }

        if (total_energy > 0) {
            bounce_distribution = DiscreteDistribution<Float>(pmf_n.data(), pmf_n.size());
            for (int i = 1; i < MAX_CHAIN_LENGTH; i++) {
                if (pmf_tau_per_bounce[i].size()) {
                    tau_distribution_per_bounce[i] =
                        DiscreteDistribution<Float>(pmf_tau_per_bounce[i].data(), pmf_tau_per_bounce[i].size());
                }
            }
            if (global_sms_config.directional_struct == 0) {
                for (const auto &[tau, pmf] : pmf_sample_for_type) {
                    if (pmf.size()) {
                        direction_distribution_for_type[tau] = DiscreteDistribution<Float>(pmf.data(), pmf.size());
                    }
                }
            } else if (global_sms_config.directional_struct == 1) {
                for (DTreeRecord &rec : drecs) {
                    dtree_for_type[rec.tau].record(rec);
                }
                for (auto &[x, y] : dtree_for_type) {
                    y.build();
                    y.reset(20, 0.01); // Threshold of [Muller et al. 2017]
                }
            }
        }

        if (global_sms_config.product_sampling) {
            for (const auto &[tau, direction_sample_for_type_tau] : direction_sample_for_type) {
                auto &cached_radius = cached_radius_for_type[tau];
                for (int i = 0; i < cached_radius.size(); i++) {
                    Vector3f dir  = direction_sample_for_type_tau[i];
                    float min_val = 1e18;
                    float eps     = 1e-6;
                    float radius2 = eps;
                    for (const Vector3f &cache_dir : direction_sample_for_type_tau) {
                        if (dir == cache_dir)
                            continue;
                        float direction_dist = dot(dir - cache_dir, dir - cache_dir);
                        min_val              = std::min(min_val, direction_dist);
                    }
                    if (min_val < 1e9) {
                        radius2 = min_val + eps;
                    }
                    cached_radius[i] = radius2;
                }
            }
        }

        perf_time_build += timer.value();
    }

    void init_product(ChainDistributionSamplingContext &ctx) {
        auto si     = ctx.si;
        Float alpha = si.bsdf()->roughness();
        std::vector<Float> pmf_tau_per_bounce[MAX_CHAIN_LENGTH];
        std::vector<Float> pmf_n(MAX_CHAIN_LENGTH);
        for (const auto &[tau, direction_sample_for_type_tau] : direction_sample_for_type) {
            const auto &cached_radius                                           = cached_radius_for_type[tau];
            const DiscreteDistribution<Float> &distribution_sample_for_type_tau = direction_distribution_for_type[tau];
            std::vector<Float> pmf_direction;
            for (int i = 0; i < direction_sample_for_type_tau.size(); i++) {
                Float value;
                Vector3f p_radiance = direction_sample_for_type_tau[i];
                BSDFContext ctx_;
                Spectrum brdf  = si.bsdf()->eval(ctx_, si, si.to_local(p_radiance));
                Float brdf_avg = (brdf[0] + brdf[1] + brdf[2]) / 3;
                value          = distribution_sample_for_type_tau.eval_pmf(i) * brdf_avg;

                float eps = 1e-6;
                if (std::isinf(value) || std::isnan(value) || value < eps)
                    value = eps;

                pmf_direction.push_back(value);
                int length = get_bit(tau);
                pmf_n[length] += value;
                pmf_tau_per_bounce[length].push_back(value);
                ctx.value_tau[length].push_back(tau);
            }
            ctx.direction_distribution_for_type[tau] =
                DiscreteDistribution<Float>(pmf_direction.data(), pmf_direction.size());
        }
        if (direction_sample_for_type.size() != 0)
            ctx.bounce_distribution = DiscreteDistribution<Float>(pmf_n.data(), pmf_n.size());
        for (int i = 0; i < MAX_CHAIN_LENGTH; i++) {
            if (pmf_tau_per_bounce[i].size()) {
                ctx.tau_distribution_per_bounce[i] =
                    DiscreteDistribution<Float>(pmf_tau_per_bounce[i].data(), pmf_tau_per_bounce[i].size());
            }
        }
    }

    bool valid_bounce(int n) {
        if (n >= MAX_CHAIN_LENGTH) {
            return false;
        }
        return value_tau[n].size() > 0;
    }

    int sample_n(ref<Sampler> sampler, const ChainDistributionSamplingContext &ctx) {
        if (!global_sms_config.product_sampling) {
            return bounce_distribution.sample(sampler->next_1d());
        } else {
            return ctx.bounce_distribution.sample(sampler->next_1d());
        }
    }

    Float pdf_n(int n, const ChainDistributionSamplingContext &ctx) {
        if (n >= MAX_CHAIN_LENGTH) {
            return 0;
        } else {
            if (!global_sms_config.product_sampling) {
                return bounce_distribution.eval_pmf_normalized(n);
            } else {
                return ctx.bounce_distribution.eval_pmf_normalized(n);
            }
        }
    }

    int sample_tau(int n, ref<Sampler> sampler, const ChainDistributionSamplingContext &ctx) {
        if (global_sms_config.ablation == 2) {
            // uniformly sample tau
            return -1;
        } else if (!global_sms_config.product_sampling) {
            return value_tau[n][tau_distribution_per_bounce[n].sample(sampler->next_1d())];
        } else {
            return ctx.value_tau[n][ctx.tau_distribution_per_bounce[n].sample(sampler->next_1d())];
        }
    }

    Vector3f noproduct_sample_omega(int tau, ref<Sampler> sampler, ChainDistributionSamplingContext &ctx) {
        Vector3f guide_dir;
        if (direction_distribution_for_type.find(tau) == direction_distribution_for_type.end()) {
            std::cout << "error no direction_distribution_for_type key " << tau << std::endl;
            return Vector3f(0., 1., 0.);
        }
        if (direction_sample_for_type.find(tau) == direction_sample_for_type.end()) {
            std::cout << "error no direction_sample_for_type key " << tau << std::endl;
            return Vector3f(0., 1., 0.);
        }
        if (cached_radius_for_type.find(tau) == cached_radius_for_type.end()) {
            std::cout << "error no cached_radius_for_type key " << tau << std::endl;
            return Vector3f(0., 1., 0.);
        }
        const DiscreteDistribution<Float> &distribution_sample_for_type_tau = direction_distribution_for_type[tau];
        const std::vector<Vector3f> &direction_sample_for_type_tau          = direction_sample_for_type[tau];
        std::vector<Float> &cached_radius                                   = cached_radius_for_type[tau];
        std::shared_ptr<std::mutex> &cached_radius_mu                       = cached_radius_for_type_mu[tau];

        int guide_idx = distribution_sample_for_type_tau.sample(sampler->next_1d());
        guide_dir     = direction_sample_for_type_tau[guide_idx];

        float radius2 = 1e-6f;
        cached_radius_mu->lock();
        if (cached_radius[guide_idx] < 0) {
            cached_radius_mu->unlock();
            float min_val = 1e18;
            float eps     = 1e-6;
            for (const Vector3f &cache_dir : direction_sample_for_type_tau) {
                if (guide_dir == cache_dir)
                    continue;
                float direction_dist = dot(guide_dir - cache_dir, guide_dir - cache_dir);
                min_val              = std::min(min_val, direction_dist);
            }
            if (min_val < 1e9) {
                radius2 = min_val + eps;
            }

            cached_radius_mu->lock();
            cached_radius[guide_idx] = radius2;
            cached_radius_mu->unlock();
        } else {
            radius2 = cached_radius[guide_idx];
            cached_radius_mu->unlock();
        }

        // Generate and sample vMF
        double k   = 1.0 / radius2;
        Point2f xi = sampler->next_2d();
        double W   = 1 + (log(xi[0] + exp(-2 * k) * (1.0 - xi[0]))) / k;
        double Wr  = sqrt(1.0 - W * W);
        Vector3f vmf_sample(Wr * sin(2 * M_PI * xi[1]), Wr * cos(2 * M_PI * xi[1]), W);

        Frame3f frame(guide_dir);
        Vector3f ogdir = guide_dir;
        guide_dir      = frame.to_world(vmf_sample);
        return guide_dir;
    }

    Vector3f product_sample_omega(int tau, ref<Sampler> sampler, ChainDistributionSamplingContext &ctx) {
        auto chain_distr = this;
        auto si          = ctx.si;
        const DiscreteDistribution<Float> &distribution_sample_for_type_tau =
            chain_distr->direction_distribution_for_type[tau];
        const std::vector<Vector3f> &direction_sample_for_type_tau = chain_distr->direction_sample_for_type[tau];
        std::vector<Float> &cached_radius                          = chain_distr->cached_radius_for_type[tau];
        DiscreteDistribution<Float> &cached_distribution           = ctx.direction_distribution_for_type[tau];
        Vector3f guide_dir;

        Float alpha     = ctx.si.bsdf()->roughness();
        double k_brdf   = 2.0 / (alpha * alpha * 4 * (abs(dot(si.n, si.wi)) + 1e-6));
        double c_brdf   = 1 / (M_PI * alpha * alpha);
        Vector3f p_brdf = normalize(2 * dot(si.n, si.wi) * si.n - si.wi);

        int guide_idx       = cached_distribution.sample(sampler->next_1d());
        Vector3f p_radiance = direction_sample_for_type_tau[guide_idx];
        double k_radiance   = 1.0 / cached_radius[guide_idx];

        Point2f xi = sampler->next_2d();
        double W   = 1 + (log(xi[0] + exp(-2 * k_radiance) * (1.0 - xi[0]))) / k_radiance;
        double Wr  = sqrt(1.0 - W * W);
        Vector3f vmf_sample(Wr * sin(2 * M_PI * xi[1]), Wr * cos(2 * M_PI * xi[1]), W);

        Frame3f frame(p_radiance);

        guide_dir = frame.to_world(vmf_sample);
        return guide_dir;
    }

    Vector3f sample_omega(int tau, ref<Sampler> sampler, ChainDistributionSamplingContext &ctx) {
        Vector3f guide_dir = Vector3f(0.0, 0.0, 0.0);
        if (global_sms_config.ablation == 3)
            return guide_dir; // If a zero-vector is returned, sampling will fail and revert to uniform direction
                              // sampling.
        NanosecondTimer timer;
        if (global_sms_config.directional_struct == 0) {
            if (global_sms_config.product_sampling && ctx.si.bsdf()->roughness() > 1e-6) {
                guide_dir = product_sample_omega(tau, sampler, ctx);
            } else {
                guide_dir = noproduct_sample_omega(tau, sampler, ctx);
            }
        } else if (global_sms_config.directional_struct == 1) {
            guide_dir = dtree_for_type[tau].sample(sampler);
        }
        perf_num_query++;
        perf_time_query += timer.value();
        return guide_dir;
    }

    Float statisticalWeight() { return weight; }

    bool valid() { return total_energy > 0; }

    void clear() {
        weight       = 0;
        total_energy = 0;
        aabb.reset();
        bounce_distribution = DiscreteDistribution<Float>();
        m_subpath_sample_list.clear();

        for (int i = 0; i < MAX_CHAIN_LENGTH; i++) {
            value_tau[i].clear();
            tau_distribution_per_bounce[i] = DiscreteDistribution<Float>();
        }

        direction_sample_for_type.clear();
        cached_radius_for_type.clear();
        cached_radius_for_type_mu.clear();
        direction_distribution_for_type.clear();
    }

    static void print_stats() {
        std::cout << "CD stats perf_time_build " << std::fixed << std::setprecision(6) << perf_time_build * 1e-9
                  << std::endl;
        std::cout << "CD stats perf_time_query " << std::fixed << std::setprecision(6) << perf_time_query * 1e-9
                  << std::endl;
        std::cout << "CD stats perf_num_query " << perf_num_query << std::endl;
        std::cout << "CD stats perf_time_query_avg " << perf_time_query * 1e-9 / perf_num_query << std::endl;
    }

    static inline std::atomic<unsigned long long> perf_time_build = 0; // nano sec
    static inline std::atomic<unsigned long long> perf_time_query = 0; // nano sec
    static inline std::atomic<unsigned long long> perf_num_query  = 0; // n

public:
    // for bounce
    DiscreteDistribution<Float> bounce_distribution;

    // for type
    DiscreteDistribution<Float> tau_distribution_per_bounce[MAX_CHAIN_LENGTH];
    std::vector<int> value_tau[MAX_CHAIN_LENGTH];

    // for direction (ours)
    mutable std::map<int, DiscreteDistribution<Float>> direction_distribution_for_type;
    mutable std::map<int, std::vector<Float>> cached_radius_for_type;
    mutable std::map<int, std::shared_ptr<std::mutex>> cached_radius_for_type_mu;
    mutable std::map<int, std::vector<Vector3f>> direction_sample_for_type;

    // for direction (dtree)
    std::vector<DTreeRecord> drecs;
    DTreeWrapper dtree;
    std::map<int, DTreeWrapper> dtree_for_type;

    std::vector<SubpathSample> m_subpath_sample_list;
    Float weight              = 0;
    Float weight_not_filtered = 0;
    Float total_energy        = 0;
    AABB aabb;

    std::mutex mu;
};

template <typename Float, typename Spectrum>
std::shared_ptr<ChainDistribution<Float, Spectrum>> chainDistributionFactory() {
    return std::make_shared<ChainDistribution<Float, Spectrum>>();
}

NAMESPACE_END(mitsuba)