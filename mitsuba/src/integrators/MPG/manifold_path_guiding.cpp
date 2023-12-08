#include "ann.h"
#include "manifold_path_guiding.h"
#include "spatial_structure.h"
#include "dtree.h"
#include "util.h"
#include "chain_distribution.h"

#define stat(n, x)                                                                                                     \
    {                                                                                                                  \
        static long long sss = 0;                                                                                      \
        sss += x;                                                                                                      \
        if (rand() % 30000 == 0) {                                                                                     \
            std::cout << n << ": " << sss << std::endl;                                                                \
        }                                                                                                              \
    }

NAMESPACE_BEGIN(mitsuba)


ManifoldPathGuidingConfig global_sms_config;


template <typename Float, typename Spectrum>
static inline ThreadLocal<std::vector<SubpathSample<Float, Spectrum>>> recorded_samples_thread_local;

// ===============================
// Manifold Sampler
// ===============================
template <typename Float, typename Spectrum> class GuidedManifoldSampler {
public:
    MTS_IMPORT_TYPES(BSDF, Sampler, Scene, Shape);
    using BSDFPtr            = typename RenderAliases::BSDFPtr;
    using ShapePtr           = typename RenderAliases::ShapePtr;
    using ManifoldVertex     = ManifoldVertex<Float, Spectrum>;
    using EmitterInteraction = EmitterInteraction<Float, Spectrum>;
    using SpecularManifold   = SpecularManifold<Float, Spectrum>;
    using Manifold_Walk      = Manifold_Walk<Float, Spectrum>;
    using SubpathSample      = SubpathSample<Float, Spectrum>;

    using SpatialStructure                 = SpatialStructure<Float, Spectrum>;
    using SpatialStructureANN              = SpatialStructureANN<Float, Spectrum>;
    using SpatialStructureSTree            = SpatialStructureSTree<Float, Spectrum>;
    using ChainDistribution                = ChainDistribution<Float, Spectrum>;
    using ChainDistributionSamplingContext = ChainDistributionSamplingContext<Float, Spectrum>;

    using MySTreeNode = MySTreeNode<Float, Spectrum>;

    int sample_bounce     = -1;
    float inv_prob_bounce = 1.0f;
    ref<Shape> sampled_shape;

    GuidedManifoldSampler() {}

    GuidedManifoldSampler(const Scene *scene, const ManifoldPathGuidingConfig &config, const SurfaceInteraction3f si,
                          const std::vector<SubpathSample> *subpath_sample_list,
                          const std::vector<SubpathSample> *subpath_sample_list_ext,
                          const std::shared_ptr<SpatialStructure> spatial_structure,
                          const std::shared_ptr<SpatialStructure> spatial_structure_ext) {
        init(scene, config, si, subpath_sample_list, subpath_sample_list_ext, spatial_structure, spatial_structure_ext);
    }

    void init(const Scene *_scene, const ManifoldPathGuidingConfig &_config, const SurfaceInteraction3f _si,
              const EmitterInteraction _ei, const std::vector<SubpathSample> *_subpath_sample_list,
              const std::vector<SubpathSample> *_subpath_sample_list_ext,
              std::shared_ptr<SpatialStructure> _spatial_structure,
              std::shared_ptr<SpatialStructure> _spatial_structure_ext) {
        si = _si, ei = _ei, m_scene = _scene, m_config = _config, manifold_walk = Manifold_Walk(_scene, _config),
        subpath_sample_list = _subpath_sample_list, subpath_sample_list_ext = _subpath_sample_list_ext,
        spatial_structure = _spatial_structure, spatial_structure_ext = _spatial_structure_ext;
        if (m_config.guided || (m_config.train_auto && m_online_iteration)) {
            NanosecondTimer timer;
            chain_distr = spatial_structure->query(si.p, ei.p);
            distr_ctx.clear();
            distr_ctx.si = si;
            if (m_config.product_sampling)
                chain_distr->init_product(distr_ctx);
            perf_time_guide += timer.value();
        }
        if (m_config.initial == 2) {
            chain_distr_ext = spatial_structure_ext->query(si.p, 0);
        }
    }

    ~GuidedManifoldSampler() {}

    std::pair<Vector3f, int> sample_seed_direction_and_tau(ref<Sampler> sampler) {
        Point3f x0         = si.p;
        bool succeed       = false;
        Vector3f guide_dir = 0;
        int tau            = -1;

        if (chain_distr && chain_distr->valid_bounce(sample_bounce) &&
            ((m_config.guided && sampler->next_1d() > m_config.prob_uniform) ||
             (m_online_iteration && !m_online_last_iteration && sampler->next_1d() > m_config.prob_uniform_train) ||
             (m_online_iteration && m_online_last_iteration && sampler->next_1d() > m_config.prob_uniform))) {
            NanosecondTimer timer;
            tau       = chain_distr->sample_tau(sample_bounce, sampler, distr_ctx);
            guide_dir = chain_distr->sample_omega(tau, sampler, distr_ctx);

            if (dot(guide_dir, si.n) > 0) {
                succeed = true;
            }
            perf_time_guide += timer.value();
        }

        // Use distribution from the external samples (e.g., photons)
        if (!succeed && m_config.initial == 2 && sampler->next_1d() > 0.5f) {
            guide_dir = chain_distr_ext->sample_omega(-1, sampler, distr_ctx);
            if (dot(guide_dir, si.n) > 0) {
                succeed = true;
            }
        }

        // Uniform
        if (!succeed) {
            if (m_config.initial != 1) {
                const std::vector<ref<Shape>> &shapes = m_scene->caustic_casters_multi_scatter();
                ref<Shape> shape_                     = shapes[sampler->next_1d() * shapes.size()];
                sampled_shape                         = shape_;
                PositionSample3f ps = shape_->sample_position(si.time, sampler->next_2d()); // ! original
                Point3f x1          = ps.p;
                guide_dir           = normalize(x1 - x0);
            } else {
                guide_dir                = warp::square_to_uniform_sphere(sampler->next_2d());
                SurfaceInteraction3f its = m_scene->ray_intersect(si.spawn_ray(guide_dir));
                sampled_shape            = (Shape *) its.shape;
                if (dot(guide_dir, si.n) < 0) {
                    guide_dir *= -1;
                }
            }
            // We wait the tau sample till the recurrent ray-tracing phase and return tau = -1.
        }
        return { guide_dir, tau };
    }

    Mask sample_path(const SurfaceInteraction3f &si, const EmitterInteraction &ei, ref<Sampler> sampler) {
        Mask success_seed = sample_seed_path(sampler);
        if (!success_seed) {
            return false;
        }
        Vector3f omega_seed     = normalize(m_current_path[0].p - si.p);
        Mask success_solve      = manifold_walk.newton_solver(si, ei, m_current_path, m_offset_normals);
        Vector3f omega_solution = normalize(m_current_path[0].p - si.p);
        if (!success_solve) {
            return false;
        }

        return true;
    }

    Mask sample_seed_path(ref<Sampler> sampler) {
        m_current_path.clear();
        SurfaceInteraction3f si_(si);
        Point3f x0     = si_.p;
        auto [wo, tau] = sample_seed_direction_and_tau(sampler);

        Ray3f ray(x0, wo, si_.time, si_.wavelengths);
        while (true) {
            int bounce = m_current_path.size();
            if (bounce >= m_seed_path.size()) {
                break;
            }
            si_ = m_scene->ray_intersect(ray);
            if (!si_.is_valid()) {
                return false;
            }
            const ShapePtr shape = si_.shape;
            if (!shape->is_caustic_caster_multi_scatter() && !shape->is_caustic_bouncer()) {
                // We intersected something that cannot be par of a specular chain
                return false;
            }

            if (shape != m_seed_path[bounce].shape) {
                /* When we sample multiple paths for the PDF estimation process,
                be conservative and only allow paths that intersect the same
                shapes again. */
                return false;
            }

            // On first intersect, make sure that we actually hit this target shape (for surface sampling)
            if (!((m_config.guided || m_config.train_auto) || m_online_iteration) && bounce == 0 &&
                (shape != sampled_shape && m_config.sms_strict_uniform)) {
                return false;
            }

            // Create the path vertex
            ManifoldVertex vertex = ManifoldVertex(si_, 0.f);
            m_current_path.push_back(vertex);
            Vector3f n_offset = m_offset_normals[bounce];

            Vector3f m           = vertex.s * n_offset[0] + vertex.t * n_offset[1] + vertex.n * n_offset[2];
            Vector3f wi          = -wo;
            bool scatter_success = false;

            float reflect_prob = 1.f;
            if (vertex.eta != 1.f) { // dielectric
                if (tau != -1) {
                    // guided type sampling
                    if (get_chaintype_bit(tau, bounce) == 1) {
                        reflect_prob = 0.0;
                    } else {
                        reflect_prob = 1.0;
                    }
                } else {
                    // Fresnel type sampling
                    auto avg              = [&](Spectrum s) { return (s[0] + s[1] + s[2]) / 3; };
                    const BSDFPtr bsdf    = si_.bsdf();
                    Complex<Spectrum> ior = si_.bsdf()->ior(si_);
                    Spectrum f;
                    Frame3f frame   = bsdf->frame(si, 0.f);
                    Float cos_theta = dot(frame.n, wi);
                    Float eta       = hmean(real(ior));
                    if (cos_theta < 0.f) {
                        eta = rcp(eta);
                    }
                    auto [F, unused_0, unused_1, unused_2] = fresnel(cos_theta, eta);
                    reflect_prob                           = avg(F);
                }
                if (m_config.sms_type_force == 1) {
                    reflect_prob = 0.0;
                }
            }
            // ! We disregard both tau and sms_type_force for conductor.

            if (sampler->next_1d() < reflect_prob) {
                std::tie(scatter_success, wo)    = SpecularManifold::reflect(wi, m);
                m_current_path.back().is_refract = false;
            } else {
                std::tie(scatter_success, wo)    = SpecularManifold::refract(wi, m, vertex.eta);
                m_current_path.back().is_refract = true;
            }
            if (!scatter_success) {
                // We must have hit total internal reflection. Abort.
                return false;
            }
            ray = si_.spawn_ray(wo);
        }

        return true;
    }

    int getTau(std::vector<ManifoldVertex> &v) {
        int tau = 0;
        set_bit(tau, sample_bounce);
        for (int i = 0; i < sample_bounce; i++) {
            set_chaintype_bit(tau, i, v[i].is_refract);
        }
        return tau;
    }

    Spectrum specular_manifold_sampling_impl(ref<Sampler> sampler) {
        Point3f x0     = si.p;
        auto [wo, tau] = sample_seed_direction_and_tau(sampler);

        Vector3f initial_direction = wo;
        Ray3f ray(x0, wo, si.time, si.wavelengths);
        SurfaceInteraction3f next_interaction = m_scene->ray_intersect(ray);
        if (!next_interaction.is_valid() || (!next_interaction.shape->is_caustic_caster_multi_scatter() &&
                                             !next_interaction.shape->is_caustic_bouncer())) {
            // We intersected something that cannot be par of a specular chain
            return 0.0;
        }

        std::vector<ManifoldVertex> sampled_path;
        m_offset_normals.clear();
        m_offset_normals_pdf  = 1.0;
        Spectrum specular_val = 0.f;
        SurfaceInteraction3f si_(si);
        while (true) {
            int bounce = sampled_path.size();
            if (bounce > sample_bounce) {
                break;
            }
            if (bounce == sample_bounce) {
                m_current_path          = sampled_path;
                Vector3f omega_seed     = normalize(m_current_path[0].p - si.p);
                Mask success            = manifold_walk.newton_solver(si, ei, m_current_path,
                                                                      m_offset_normals); // Newton Solver will change current_path
                Vector3f omega_solution = normalize(m_current_path[0].p - si.p);

                ManifoldVertex vtx_last = m_current_path[m_current_path.size() - 1];
                auto [success_e, vy] =
                    SpecularManifold::emitter_interaction_to_vertex(m_scene, ei, vtx_last.p, si.time, si.wavelengths);
                if (success && success_e) {
                    m_seed_path                    = m_current_path;
                    Point3f x1                     = m_current_path[0].p;
                    Vector3f direction             = normalize(m_current_path[0].p - si.p);
                    std::vector<ManifoldVertex> &v = m_current_path;
                    int sampled_tau                = getTau(m_current_path);

                    // * EVAL
                    NanosecondTimer timer_eval;
                    Spectrum path_contribution =
                        SpecularManifold::evaluate_path_contribution(m_scene, m_current_path, si, ei);
                    perf_time_eval += timer_eval.value();
                    BSDFContext ctx;
                    Spectrum bsdf_val       = si.bsdf()->eval(ctx, si, si.to_local(direction));
                    int iterations          = 1;
                    Float inv_prob_estimate = 0.0;
                    int repeat_bernoulli    = global_sms_config.rep_ber;
                    if (repeat_bernoulli < 0) {
                        // means repeating during the rendering phase only
                        repeat_bernoulli *= -1;
                        if (m_online_last_iteration == false) {
                            repeat_bernoulli = 1;
                        }
                    }
                    NanosecondTimer timer;
                    for (int _ = 0; _ < repeat_bernoulli; _++) {
                        inv_prob_estimate += 1;
                        while (true) {
                            ScopedPhase scope_phase(ProfilerPhase::SMSCausticsBernoulliTrials);
                            bool success_trial = sample_path(si, ei, sampler);

                            if (success_trial) {
                                Vector3f direction_trial = normalize(m_current_path[0].p - si.p);
                                int trial_tau            = getTau(m_current_path);

                                if (trial_tau == sampled_tau &&
                                    abs(dot(direction, direction_trial) - 1.f) < m_config.uniqueness_threshold) {
                                    break;
                                }
                            }
                            inv_prob_estimate += 1.f;
                            iterations++;

                            if (m_config.max_trials > 0 && iterations > m_config.max_trials * repeat_bernoulli) {
                                /* There is a tiny chance always to sample super
                                weird paths that will never occur again due to small
                                numerical imprecisions. So setting a (super
                                conservative) threshold here can help to avoid
                                infinite loops. */
                                inv_prob_estimate = 0.f;
                                break;
                            }
                        }
                    }
                    inv_prob_estimate /= repeat_bernoulli;
                    perf_time_bernoulli += timer.value();
                    specular_val += bsdf_val * path_contribution * inv_prob_estimate / m_offset_normals_pdf;

                    // * STORE

                    Spectrum result = path_contribution * inv_prob_estimate / m_offset_normals_pdf;
                    auto avg        = [&](Spectrum s) { return (s[0] + s[1] + s[2]) / 3; };

                    Float energy = avg(result);

                    int tau = 0;
                    set_bit(tau, sample_bounce);
                    for (int i = 0; i < sample_bounce; i++) {
                        set_chaintype_bit(tau, i, v[i].is_refract);
                    }
                    float eps = 1e-6;

                    if (energy > eps) {
                        SubpathSample new_cache_info;
                        new_cache_info.xD     = si.p;
                        new_cache_info.xL     = ei.p;
                        new_cache_info.x1     = x1;
                        new_cache_info.energy = energy * inv_prob_bounce;
                        new_cache_info.bounce = bounce;
                        new_cache_info.type   = tau;
                        std::vector<SubpathSample> &new_data =
                            (std::vector<SubpathSample> &) recorded_samples_thread_local<Float, Spectrum>;
                        new_data.push_back(new_cache_info);
                    }
                }
            }

            si_ = m_scene->ray_intersect(ray);
            if (!si_.is_valid()) {
                break;
            }
            const ShapePtr shape = si_.shape;
            if (!shape->is_caustic_caster_multi_scatter() && !shape->is_caustic_bouncer()) {
                // We intersected something that cannot be par of a specular chain
                break;
            }

            // On first intersect, make sure that we actually hit this target shape (for surface sampling).
            if (!(m_config.guided || m_online_iteration) && bounce == 0 &&
                (shape != sampled_shape && m_config.sms_strict_uniform)) {
                return false;
            }

            // Create the path vertex
            ManifoldVertex vertex = ManifoldVertex(si_, 0.f);
            sampled_path.push_back(vertex);

            // Potentially sample an offset normal here.
            Vector3f n_offset;
            n_offset    = Vector3f(0.f, 0.f, 1.f);
            Float p_o   = 1.f;
            Float alpha = shape->bsdf()->roughness();
            if (alpha > 0.f) {
                // Assume isotropic Beckmann distribution for the surface
                // roughness.
                MicrofacetDistribution<Float, Spectrum> distr(MicrofacetType::Beckmann, alpha, alpha, false);
                std::tie(n_offset, p_o) = distr.sample(Vector3f(0.f, 0.f, 1.f), sampler->next_2d());
            }
            m_offset_normals_pdf *= p_o;
            m_offset_normals.push_back(n_offset);

            // Perform scattering at vertex, unless we are doing the
            // straight-line MNEE initialization
            Vector3f m = vertex.s * n_offset[0] + vertex.t * n_offset[1] + vertex.n * n_offset[2];

            Vector3f wi          = -wo;
            bool scatter_success = false;

            float reflect_prob = 1.f;
            if (vertex.eta != 1.f) { // dielectric
                if (tau != -1) {
                    if (get_chaintype_bit(tau, bounce) == 1) {
                        reflect_prob = 0.0;
                    } else {
                        reflect_prob = 1.0;
                    }
                } else {
                    auto avg              = [&](Spectrum s) { return (s[0] + s[1] + s[2]) / 3; };
                    const BSDFPtr bsdf    = si_.bsdf();
                    Complex<Spectrum> ior = si_.bsdf()->ior(si_);
                    Spectrum f;
                    Frame3f frame   = bsdf->frame(si, 0.f);
                    Float cos_theta = dot(frame.n, wi);
                    Float eta       = hmean(real(ior));
                    if (cos_theta < 0.f) {
                        eta = rcp(eta);
                    }
                    auto [F, unused_0, unused_1, unused_2] = fresnel(cos_theta, eta);
                    reflect_prob                           = avg(F);
                }
                if (m_config.sms_type_force == 1) {
                    reflect_prob = 0.0;
                }
            }

            if (sampler->next_1d() < reflect_prob) {
                std::tie(scatter_success, wo)  = SpecularManifold::reflect(wi, m);
                sampled_path.back().is_refract = false;
            } else {
                std::tie(scatter_success, wo)  = SpecularManifold::refract(wi, m, vertex.eta);
                sampled_path.back().is_refract = true;
            }

            if (!scatter_success) {
                // We must have hit total internal reflection. Abort.
                break;
            }
            ray = si_.spawn_ray(wo);
        }
        return specular_val;
    }

    Spectrum specular_manifold_sampling(ref<Sampler> sampler) {
        if (unlikely(!si.is_valid())) {
            return 0.f;
        }
        NanosecondTimer timer;

        auto shapes = m_scene->caustic_casters_multi_scatter();
        if (unlikely(shapes.size() == 0)) {
            return 0.f;
        }

        DiscreteDistribution<Float> geometric_distr;
        std::vector<Float> a;
        a.resize(MAX_CHAIN_LENGTH);
        float acc = 1.0f;
        for (int i = 0; i < MAX_CHAIN_LENGTH; i++) {
            a[i] = acc;
            if (i + 1 > m_sms_rr_depth) {
                acc *= 0.95; // hard coded q
            }
            if (i + 1 > m_sms_max_depth) {
                break;
            }
        }
        geometric_distr = DiscreteDistribution<Float>(a.data(), a.size());

        auto sample_geometric_distr = [&](float r) -> int { return geometric_distr.sample(r); };
        auto pdf_geometric_distr    = [&](int x) -> float { return geometric_distr.eval_pmf_normalized(x); };

        float bounce_alpha = 0.5;

        if (m_config.ablation == 1) {
            bounce_alpha = 1.0;
        }

        if (m_config.sms_bounce_force > 0) {
            sample_bounce   = m_config.sms_bounce_force;
            inv_prob_bounce = 1.0;
        } else if ((m_config.guided || (m_config.train_auto && m_online_iteration > 0)) == false ||
                   !chain_distr->valid()) {
            sample_bounce   = sample_geometric_distr(sampler->next_1d()) + 1;
            inv_prob_bounce = 1.0 / pdf_geometric_distr(sample_bounce - 1);
        } else {
            if (sampler->next_1d() < bounce_alpha) {
                sample_bounce = sample_geometric_distr(sampler->next_1d()) + 1;
            } else {
                sample_bounce = chain_distr->sample_n(sampler, distr_ctx);
            }
            inv_prob_bounce = 1.0 / ((pdf_geometric_distr(sample_bounce - 1) * bounce_alpha +
                                      chain_distr->pdf_n(sample_bounce, distr_ctx) * (1 - bounce_alpha)));
        }

        auto ans = specular_manifold_sampling_impl(sampler) * inv_prob_bounce;
        perf_time_all += timer.value();
        return ans;
    }

    static void print_stats() {
        std::cout << "GuidedManifoldSampler stats perf_time_guide " << std::fixed << std::setprecision(6)
                  << perf_time_guide * 1e-9 << std::endl;
        std::cout << "GuidedManifoldSampler stats perf_time_bernoulli " << std::fixed << std::setprecision(6)
                  << perf_time_bernoulli * 1e-9 << std::endl;
        std::cout << "GuidedManifoldSampler stats perf_time_eval " << std::fixed << std::setprecision(6)
                  << perf_time_eval * 1e-9 << std::endl;
        std::cout << "GuidedManifoldSampler stats perf_time_all " << std::fixed << std::setprecision(6)
                  << perf_time_all * 1e-9 << std::endl;
    }

    SurfaceInteraction3f si;
    EmitterInteraction ei;
    const Scene *m_scene = nullptr;
    ManifoldPathGuidingConfig m_config;
    std::vector<ManifoldVertex> m_seed_path, m_current_path;
    std::vector<Vector3f> m_offset_normals;
    Manifold_Walk manifold_walk;
    Float m_offset_normals_pdf;
    const std::vector<SubpathSample> *subpath_sample_list     = nullptr;
    const std::vector<SubpathSample> *subpath_sample_list_ext = nullptr;
    std::shared_ptr<SpatialStructure> spatial_structure       = nullptr;
    std::shared_ptr<SpatialStructure> spatial_structure_ext   = nullptr;
    int m_online_iteration                                    = 0;
    bool m_online_last_iteration                              = 0;
    int m_sms_rr_depth                                        = 0;
    int m_sms_max_depth                                       = 0;

    mutable ChainDistributionSamplingContext distr_ctx;

    static inline std::atomic<unsigned long long> perf_time_guide     = 0; // nano sec
    static inline std::atomic<unsigned long long> perf_time_bernoulli = 0; // nano sec
    static inline std::atomic<unsigned long long> perf_time_eval      = 0; // nano sec
    static inline std::atomic<unsigned long long> perf_time_all       = 0; // nano sec

    std::shared_ptr<ChainDistribution> chain_distr;
    std::shared_ptr<ChainDistribution> chain_distr_ext;
};

// ===============================
// Integrator
// ===============================
template <typename Float, typename Spectrum>
class ManifoldPathGuidingIntegrator : public MonteCarloIntegrator<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(MonteCarloIntegrator, m_max_depth, m_rr_depth);
    MTS_IMPORT_TYPES(Scene, Sampler, Sensor, Emitter, EmitterPtr, BSDF, BSDFPtr, ShapePtr, Medium);
    using SubpathSample         = SubpathSample<Float, Spectrum>;
    using GuidedManifoldSampler = GuidedManifoldSampler<Float, Spectrum>;

    using SpatialStructure      = SpatialStructure<Float, Spectrum>;
    using SpatialStructureANN   = SpatialStructureANN<Float, Spectrum>;
    using SpatialStructureSTree = SpatialStructureSTree<Float, Spectrum>;
    using ChainDistribution     = ChainDistribution<Float, Spectrum>;

    using MySTreeNode = MySTreeNode<Float, Spectrum>;

    ManifoldPathGuidingIntegrator(const Properties &props) : Base(props) {
        m_sms_config = ManifoldPathGuidingConfig();

        // core
        m_sms_config.guided     = props.bool_("guided", false); // guide or not
        m_sms_config.train_auto = props.bool_("train_auto", false);

        // initialization
        m_sms_config.initial = props.int_("initial", 0);

        // ablation
        m_sms_config.ablation = props.int_("ablation", 0);

        // spatial struct
        m_sms_config.spatial_struct = props.int_("spatial_struct", 1);
        m_sms_config.spatial_filter = props.float_("spatial_filter", 0.1f);
        m_sms_config.knn_k          = props.int_("knn_k", -1);

        // directional struct
        m_sms_config.directional_struct = props.int_("directional_struct", 0);

        // online training related
        m_sms_config.budget_is_time = props.bool_("budget_is_time", true);
        m_sms_config.train_budget   = props.float_("train_budget", 0.3f);
        m_sms_config.train_fusion   = props.bool_("train_fusion", false);

        // utils
        m_sms_config.selective          = props.bool_("selective", false);
        m_sms_config.sms_strict_uniform = props.bool_("sms_strict_uniform", false);
        m_sms_config.sms_bounce_force   = props.int_("sms_bounce_force", 0);
        m_sms_config.sms_type_force     = props.int_("sms_type_force", 0);
        m_sms_config.seedid             = props.int_("seedid", 2023);
        m_sms_config.rep_ber            = props.int_("rep_ber", 1);

        // product sampling
        m_sms_config.product_sampling = props.bool_("product_sampling", false);

        // sms core
        m_sms_config.solver_threshold     = props.float_("solver_threshold", 1e-4f);
        m_sms_config.uniqueness_threshold = props.float_("uniqueness_threshold", 1e-4f);
        m_sms_config.max_trials           = props.int_("max_trials", 1e6);

        // sms legacy
        m_sms_config.halfvector_constraints = props.bool_("halfvector_constraints", false);
        m_sms_config.step_scale             = props.float_("step_scale", 1.f);
        m_sms_config.max_iterations         = props.int_("max_iterations", 20);

        // legacy
        m_sms_config.prob_uniform_train = props.float_("prob_uniform_train", 0.5f);
        m_sms_config.prob_uniform       = props.float_("prob_uniform", 0.5f);
        m_sms_config.sms_max_depth      = props.int_("sms_max_depth", 99);

        global_sms_config = m_sms_config;
    }

    ~ManifoldPathGuidingIntegrator() {
        delete subpath_sample_list;
        delete subpath_sample_list_ext;
    }

    bool render(Scene *scene, Sensor *sensor) override {
        // load cache data here
        Timer my_timer;

        subpath_sample_list     = new std::vector<SubpathSample>();
        subpath_sample_list_ext = new std::vector<SubpathSample>();
        spatial_structure       = spatialStructureFactory<Float, Spectrum>(scene->bbox());
        spatial_structure_ext   = spatialStructureFactory<Float, Spectrum>(scene->bbox());

        if (m_sms_config.initial == 2) {
            std::cout << "Loading from external recorded samples..." << std::endl;
            std::ifstream in("my_log2.txt");
            std::string line;
            while (std::getline(in, line)) {
                std::stringstream ss(line);
                Point3f xD   = 0;
                Point3f xL   = 0;
                Point3f x1   = 0;
                Float energy = 0;
                int bounce   = 0;
                int tau      = -1;
                float t;
                ss >> xD[0] >> xD[1] >> xD[2] >>
                    xL[0] >> xL[1] >> xL[2] >>
                    x1[0] >> x1[1] >> x1[2] >>
                    energy >> bounce;
                SubpathSample input_cache = { xD, xL, x1, energy, bounce, tau };
                subpath_sample_list_ext->push_back(input_cache);
            }
            std::cout << "Loaded sample num " << subpath_sample_list_ext->size() << std::endl;
            spatial_structure_ext->build(*subpath_sample_list_ext, m_sms_config.knn_k);
        }

        bool result;
        if (m_sms_config.train_auto == false) {
            result = MonteCarloIntegrator<Float, Spectrum>::render(scene, sensor);
        } else {
            float total_time = m_timeout;
            if (total_time < 0) {
                total_time = 1e9;
            }
            float total_spp = scene->sensors()[0]->sampler()->m_sample_count;
            int round_id = 0;
            int used_spp = 0;

            while (true) {
                std::cout << "Round " << round_id << std::endl;

                float used_time = my_timer.value() * 0.001f;


                // The following code gets information for this iteration

                // Training strategy
                bool is_last_iter  = false;
                bool need_rebuild  = true;
                int this_iter_spp  = 0;
                int this_iter_time = 0;

                if (m_sms_config.budget_is_time) {
                    if (used_time >= total_time * m_sms_config.train_budget) {
                        // Next iteration is rendering
                        need_rebuild   = false;
                        is_last_iter   = true;
                        this_iter_time = total_time - used_time;
                        this_iter_spp  = 999999;
                    } else if (used_time > total_time * m_sms_config.train_budget * 0.5f) {
                        // Next iteration is training (continue this iteration,
                        // do not delete samples)
                        need_rebuild   = false;
                        this_iter_time = total_time * m_sms_config.train_budget - used_time;
                        this_iter_spp  = 999999;
                    } else {
                        // Next iteration is training (new iteration)
                        need_rebuild   = true;
                        this_iter_time = total_time * m_sms_config.train_budget - used_time;
                        this_iter_spp  = (1 << round_id);
                    }
                } else {
                    this_iter_time = 1e9;
                    if (used_spp >= total_spp * m_sms_config.train_budget) {
                        // Next iteration is rendering
                        need_rebuild  = false;
                        is_last_iter  = true;
                        this_iter_spp = total_spp - used_spp;
                    } else if (used_spp > total_spp * m_sms_config.train_budget * 0.5f) {
                        // Next iteration is training (continue this iteration,
                        // do not delete samples)
                        need_rebuild  = false;
                        this_iter_spp = total_spp * m_sms_config.train_budget - used_spp;
                    } else {
                        // Next iteration is training (new iteration)
                        need_rebuild  = true;
                        this_iter_spp = (1 << round_id);
                    }
                }

                this_iter_time = std::max(1, this_iter_time);
                this_iter_spp  = std::max(1, this_iter_spp);

                m_online_iteration                             = round_id;
                g_iter                                         = m_online_iteration;
                m_online_last_iteration                        = is_last_iter;
                scene->sensors()[0]->sampler()->m_sample_count = this_iter_spp;
                m_timeout                                      = this_iter_time;
                used_spp += this_iter_spp;

                if (need_rebuild || is_last_iter) {
                    if (round_id > 0) {
                        *subpath_sample_list = global_new_data;

                        if (need_rebuild && m_sms_config.train_fusion == false) {
                            global_new_data.clear();
                        }

                        std::cout << "Recorded " << subpath_sample_list->size() << " samples" << std::endl;

                        if (global_sms_config.spatial_struct != 2) { // 2 means naive sdtree
                            spatial_structure = spatialStructureFactory<Float, Spectrum>(scene->bbox());
                        }

                        spatial_structure->build(*subpath_sample_list, m_sms_config.knn_k);
                    }
                }
                    
                result = MonteCarloIntegrator<Float, Spectrum>::render(scene, sensor);

                if (is_last_iter) {
                    break;
                }

                round_id++;
            }
        }

        spatial_structure->print_stats();
        ChainDistribution::print_stats();
        GuidedManifoldSampler::print_stats();
        print_stats();
        return result;
    }

    void render_block(const Scene *scene, const Sensor *sensor, Sampler *sampler, ImageBlock *block, Float *aovs,
                      size_t sample_count_) const override {
        MonteCarloIntegrator<Float, Spectrum>::render_block(scene, sensor, sampler, block, aovs, sample_count_);

        // Record samples for each block
        if (m_sms_config.train_auto && !m_online_last_iteration) {
            static std::mutex mu;
            mu.lock();
            std::vector<SubpathSample> &new_data =
                (std::vector<SubpathSample> &) recorded_samples_thread_local<Float, Spectrum>;
            for (const SubpathSample &cache : new_data)
                global_new_data.push_back(cache);
            new_data.clear();
            mu.unlock();
        }
    }

    std::pair<Spectrum, Mask> sample(const Scene *scene, Sampler *sampler, const RayDifferential3f &ray_,
                                     const Medium * /* medium */, Float * /* aovs */, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        GuidedManifoldSampler &mf = (GuidedManifoldSampler &) thread_mf;
        if constexpr (is_array_v<Float>) {
            Throw("This integrator does not support vector/gpu/autodiff modes!");
            return { 0.f, 0.f };
        } else {
            RayDifferential3f ray = ray_;
            Float eta             = 1.f;
            Spectrum throughput(1.f), result(0.f);
            bool specular_camera_path = true; // To capture emitters visible direcly through purely
                                              // specular reflection/refractions
            bool still_perform_nee = true;

            // ---------------------- First intersection ----------------------

            SurfaceInteraction3f si = scene->ray_intersect(ray);
            Mask valid_ray          = si.is_valid();
            EmitterPtr emitter      = si.emitter(scene);

            if (emitter) {
                result += emitter->eval(si);
            }

            // ---------------------- Main loop ----------------------

            for (int depth = 1;; ++depth) {

                // ------------------ Possibly terminate path -----------------

                if (!si.is_valid())
                    break;
                si.compute_partials(ray);

                if (depth > m_rr_depth) {
                    Float q = min(hmax(depolarize(throughput)) * sqr(eta), .95f);
                    if (sampler->next_1d() > q)
                        break;
                    throughput *= rcp(q);
                }

                if (uint32_t(depth) >= uint32_t(m_max_depth))
                    break;

                // --------------- Specular Manifold Sampling -----------------
                bool on_caustic_caster = si.shape->is_caustic_caster_multi_scatter() || si.shape->is_caustic_bouncer();

                if (!on_caustic_caster) {
                    still_perform_nee = true; // this is a new separator, reset flag
                }

                if (si.shape->is_caustic_receiver() && !on_caustic_caster &&
                    (m_max_depth < 0 || depth + 1 < m_max_depth)) {
                    EmitterInteraction ei = SpecularManifold<Float, Spectrum>::sample_emitter_interaction(
                        si, scene->caustic_emitters_multi_scatter(), sampler);

                    mf.m_online_iteration      = m_online_iteration;
                    mf.m_online_last_iteration = m_online_last_iteration;
                    mf.m_sms_max_depth         = std::max(1, m_max_depth - depth);
                    mf.m_sms_rr_depth          = std::max(1, m_rr_depth - depth);

                    mf.init(scene, m_sms_config, si, ei, subpath_sample_list, subpath_sample_list_ext,
                            spatial_structure, spatial_structure_ext);

                    perf_manifold_query++;

                    // This is where we trigger manifold sampling
                    // For selective activation, we only enable manifold sampling when there is at least one sample nearby
                    // If we enable manifold sampling, we no longer enable NEE for caustic casters
                    if (!(m_sms_config.guided || m_sms_config.train_auto) || !mf.chain_distr ||
                        (mf.chain_distr && !(mf.chain_distr->weight_not_filtered < 1 && m_sms_config.selective &&
                                             m_online_last_iteration))) {
                        perf_manifold_executed++;
                        still_perform_nee = false;
                        result += throughput * mf.specular_manifold_sampling(sampler);
                    }
                }


                // The following codes are mostly from original SMS
                // --------------------- Emitter sampling ---------------------
                BSDFContext ctx;
                ctx.sampler  = sampler;
                BSDFPtr bsdf = si.bsdf(ray);
                if ((has_flag(bsdf->flags(), BSDFFlags::Smooth) && !on_caustic_caster) || still_perform_nee) {

                    auto [ds, emitter_weight] = scene->sample_emitter_direction(si, sampler->next_2d(), true);
                    if (ds.pdf != 0.f) {
                        Vector3f wo       = si.to_local(ds.d);
                        Spectrum bsdf_val = bsdf->eval(ctx, si, wo);
                        bsdf_val          = si.to_world_mueller(bsdf_val, -wo, si.wi);
                        Float bsdf_pdf = bsdf->pdf(ctx, si, wo);
                        Float mis      = select(ds.delta, 1.f, mis_weight(ds.pdf, bsdf_pdf));
                        result += mis * throughput * bsdf_val * emitter_weight;
                    }
                }
                // ----------------------- BSDF sampling ----------------------
                auto [bs, bsdf_weight] = bsdf->sample(ctx, si, sampler->next_1d(), sampler->next_2d());
                bsdf_weight            = si.to_world_mueller(bsdf_weight, -bs.wo, si.wi);

                throughput = throughput * bsdf_weight;
                eta *= bs.eta;
                if (!has_flag(bs.sampled_type, BSDFFlags::Delta)) {
                    specular_camera_path = false;
                }
                if (all(eq(throughput, 0.f)))
                    break;

                ray                          = si.spawn_ray(si.to_world(bs.wo));
                SurfaceInteraction3f si_bsdf = scene->ray_intersect(ray);
                emitter                      = si_bsdf.emitter(scene);

                // Hit emitter after BSDF sampling
                if (emitter) {
                    if (!on_caustic_caster || (!specular_camera_path && still_perform_nee) ||
                        emitter->is_environment()) { // filter out glints
                        Spectrum emitter_val = emitter->eval(si_bsdf);
                        DirectionSample3f ds(si_bsdf, si);
                        ds.object         = emitter;
                        Float emitter_pdf = select(!has_flag(bs.sampled_type, BSDFFlags::Delta),
                                                   scene->pdf_emitter_direction(si, ds), 0.f);
                        Float mis         = mis_weight(bs.pdf, emitter_pdf);
                        result += mis * throughput * emitter_val;
                    }
                }
                si = std::move(si_bsdf);
            }

            return { result, valid_ray };
        }
    }

    Float mis_weight(Float pdf_a, Float pdf_b) const {
        pdf_a *= pdf_a;
        pdf_b *= pdf_b;
        return select(pdf_a > 0.f, pdf_a / (pdf_a + pdf_b), 0.f);
    }

    void print_stats() {
        std::cout << "Manifold perf_manifold_query " << perf_manifold_query << std::endl;
        std::cout << "Manifold perf_manifold_executed " << perf_manifold_executed << std::endl;
        std::cout << "Manifold perf_manifold_executed_ratio " << perf_manifold_executed * 1.f / perf_manifold_query
                  << std::endl;
    }

    MTS_DECLARE_CLASS()

public:
    ManifoldPathGuidingConfig m_sms_config;
    std::vector<SubpathSample> *subpath_sample_list;
    std::vector<SubpathSample> *subpath_sample_list_ext;
    std::shared_ptr<SpatialStructure> spatial_structure;
    std::shared_ptr<SpatialStructure> spatial_structure_ext;
    mutable std::vector<SubpathSample> global_new_data;
    static inline ThreadLocal<GuidedManifoldSampler> thread_mf;
    int m_online_iteration       = 0;
    bool m_online_last_iteration = 0;

    static inline std::atomic<unsigned long long> perf_manifold_executed = 0; // nano sec
    static inline std::atomic<unsigned long long> perf_manifold_query    = 0; // nano sec
};

MTS_IMPLEMENT_CLASS_VARIANT(ManifoldPathGuidingIntegrator, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(ManifoldPathGuidingIntegrator, "manifold path guiding integrator");
NAMESPACE_END(mitsuba)