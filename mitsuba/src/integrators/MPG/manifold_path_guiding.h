#pragma once

#include <mitsuba/core/fwd.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/fwd.h>
#include "util.h"

NAMESPACE_BEGIN(mitsuba)

struct ManifoldPathGuidingConfig {
    // core
    bool guided     = false;
    bool train_auto = false;

    // ablation
    int ablation = 0; // 0-original, 1-bounce, 2-tau, 3-direction

    // initialization
    int initial      = 0; // 0-surface 1-direction 2-photon
    int sms_rr_depth = 5; // initial bounce sampling uniform limit

    // spatial structure
    int spatial_struct   = 1;    // 0-knn 1-stree
    float spatial_filter = 0.1f; // overlap
    int knn_k            = -1;   // actually legacy, but may be used

    // directional struct
    int directional_struct = 0; // 0-pf 1-quadtree

    // online training related
    bool budget_is_time = true;
    float train_budget  = 0.3f;
    bool train_fusion   = false;

    // product sampling
    bool product_sampling = false;

    // utils
    bool selective          = false;
    bool naive              = false;
    int sms_bounce_force    = 0;     // if >0, disable bounce sampling
    int sms_type_force      = 0;     // if >0, disable type sampling
    bool sms_strict_uniform = false; // make sure we actually hit the sampled shape for surface sampling
    int seedid              = 0;

    // sms core
    float solver_threshold     = 1e-4f; // Newton solver stopping criterion
    float uniqueness_threshold = 1e-4f; // Threshold to distinguish unique solution paths
    int max_trials             = -1;    // Bernoulli trials max (for unbiased SMS)
    int rep_ber                = 1;     // Bernoulli trials max (for unbiased SMS)

    // legacy
    int sms_max_depth           = 99; // max depth for bounce sampling
    float prob_uniform_train    = 0.5f;
    float prob_uniform          = 0.1f;
    bool halfvector_constraints = false; // Switch back to original half-vector based constraints?

    float step_scale = 1.f;     // Scale step sizes inside Newton solver (mostly for
                                // visualizations)
    size_t max_iterations = 20; // Maxiumum number of allowed iterations of the Newton solver

    ManifoldPathGuidingConfig() {}

    std::string to_string() const {
        std::ostringstream oss;
        oss << "ManifoldPathGuidingConfig[]";
        return oss.str();
    }
};

template <typename Float_, typename Spectrum_> struct ManifoldVertex {
    using Float    = Float_;
    using Spectrum = Spectrum_;
    MTS_IMPORT_RENDER_BASIC_TYPES()
    using ShapePtr             = typename RenderAliases::ShapePtr;
    using SurfaceInteraction3f = typename RenderAliases::SurfaceInteraction3f;

    // Position and partials
    Point3f p;
    Vector3f dp_du, dp_dv;

    // Normal and partials
    Normal3f n, gn;
    Vector3f dn_du, dn_dv;
    bool is_refract;

    // Tangents and partials
    Vector3f s, t;
    Vector3f ds_du, ds_dv;
    Vector3f dt_du, dt_dv;

    // Further information
    Float eta;
    Vector2f uv;
    ShapePtr shape;
    Mask fixed_direction;

    // Used in multi-bounce version
    Vector2f C;
    Matrix2f dC_dx_prev, dC_dx_cur, dC_dx_next;
    Matrix2f tmp, inv_lambda;
    Vector2f dx;

    ManifoldVertex(const Point3f &p = Point3f(0.f))
        : p(p), dp_du(0.f), dp_dv(0.f), n(0.f), gn(0.f), dn_du(0.f), dn_dv(0.f), s(0.f), t(0.f), ds_du(0.f), ds_dv(0.f),
          dt_du(0.f), dt_dv(0.f), eta(1.f), uv(0.f), shape(nullptr), fixed_direction(false) {}

    ManifoldVertex(const SurfaceInteraction3f &si, Float smoothing = 0.f)
        : p(si.p), dp_du(si.dp_du), dp_dv(si.dp_dv), gn(si.n), uv(si.uv), shape(si.shape), fixed_direction(false) {

        // Encode conductors with eta=1.0, and dielectrics with their relative
        // IOR
        Complex<Spectrum> ior = si.bsdf()->ior(si);
        eta                   = select(all(eq(0.f, imag(ior))), hmean(real(ior)),
                                       1.f); // Assumption here is that real (dielectric) IOR is
                                             // not spectrally varying.

        // Compute frame and its derivative
        Frame3f frame = si.bsdf()->frame(si, smoothing);
        n             = frame.n;
        s             = frame.s;
        t             = frame.t;

        auto [dframe_du, dframe_dv] = si.bsdf()->frame_derivative(si, smoothing);
        dn_du                       = dframe_du.n;
        dn_dv                       = dframe_dv.n;
        ds_du                       = dframe_du.s;
        ds_dv                       = dframe_dv.s;
        dt_du                       = dframe_du.t;
        dt_dv                       = dframe_dv.t;

        // In rare cases, e.g. 'twosided' materials, the geometric normal needs
        // to be flipped
        masked(gn, dot(n, gn) < 0.f) *= -1.f;
    }

    void make_orthonormal() {
        // Turn into orthonormal parameterization at 'p'
        Float inv_norm = rcp(norm(dp_du));
        dp_du *= inv_norm;
        dn_du *= inv_norm;
        Float dp           = dot(dp_du, dp_dv);
        Vector3f dp_dv_tmp = dp_dv - dp * dp_du;
        Vector3f dn_dv_tmp = dn_dv - dp * dn_du;
        inv_norm           = rcp(norm(dp_dv_tmp));
        dp_dv              = dp_dv_tmp * inv_norm;
        dn_dv              = dn_dv_tmp * inv_norm;
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "ManifoldVertex[" << std::endl
            << "  p = " << p << "," << std::endl
            << "  n = " << n << "," << std::endl
            << "  gn = " << gn << "," << std::endl
            << "  dp_du = " << dp_du << "," << std::endl
            << "  dp_dv = " << dp_dv << "," << std::endl
            << "  dn_du = " << dn_du << "," << std::endl
            << "  dn_dv = " << dn_dv << "," << std::endl
            << "  eta = " << eta << "," << std::endl
            << "  uv = " << uv << std::endl
            << "]";
        return oss.str();
    }
};

template <typename Float_, typename Spectrum_> struct EmitterInteraction {
    using Float    = Float_;
    using Spectrum = Spectrum_;
    MTS_IMPORT_RENDER_BASIC_TYPES()
    using EmitterPtr = typename RenderAliases::EmitterPtr;

    Point3f p; // Emitter position (for area / point)
    Normal3f n;
    Vector3f d; // Emitter direction (for infinite / directional )
    Point2f uv     = 0.0f;
    Vector3f dp_du = 0.0f, dp_dv = 0.0f;
    Spectrum weight; // Samping weight (already divided by positional sampling
                     // pdf)
    Float pdf;       // Sampling pdf

    EmitterPtr emitter = nullptr;

    bool is_point() const { return has_flag(emitter->flags(), EmitterFlags::DeltaPosition); }

    bool is_directional() const {
        return has_flag(emitter->flags(), EmitterFlags::DeltaDirection) ||
               has_flag(emitter->flags(), EmitterFlags::Infinite);
    }

    bool is_area() const { return has_flag(emitter->flags(), EmitterFlags::Surface); }

    bool is_delta() const {
        return has_flag(emitter->flags(), EmitterFlags::DeltaPosition) ||
               has_flag(emitter->flags(), EmitterFlags::DeltaDirection);
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "EmitterInteraction[" << std::endl
            << "  p = " << p << "," << std::endl
            << "  n = " << n << "," << std::endl
            << "  d = " << d << "," << std::endl
            << "  weight = " << weight << "," << std::endl
            << "  pdf    = " << pdf << "," << std::endl
            << "]";
        return oss.str();
    }
};

template <typename Float_, typename Spectrum_> struct SpecularManifold {
    using Float    = Float_;
    using Spectrum = Spectrum_;
    MTS_IMPORT_TYPES(Scene, Sampler, Sensor, Emitter, EmitterPtr, BSDF, BSDFPtr, ShapePtr, Medium);
    using EmitterPtr         = typename RenderAliases::EmitterPtr;
    using ShapePtr           = typename RenderAliases::ShapePtr;
    using ManifoldVertex     = ManifoldVertex<Float, Spectrum>;
    using EmitterInteraction = EmitterInteraction<Float, Spectrum>;

    /// Sample emitter interaction for specular manifold sampling
    static EmitterInteraction sample_emitter_interaction(const SurfaceInteraction3f &si,
                                                         const std::vector<ref<Emitter>> emitters,
                                                         ref<Sampler> sampler) {
        EmitterInteraction ei;
        Spectrum spec = 0.f;

        if (unlikely(emitters.empty())) {
            Log(Warn, "Specular manifold sampling: no emitter is marked!");
            return ei;
        }

        // Uniformly sample an emitter, same was as in
        // Scene::sample_emitter_direction
        Float emitter_sample = sampler->next_1d();
        Float emitter_pdf    = 1.f / emitters.size();
        UInt32 index = min(UInt32(emitter_sample * (ScalarFloat) emitters.size()), (uint32_t) emitters.size() - 1);
        const EmitterPtr emitter = gather<EmitterPtr>(emitters.data(), index);
        ei.emitter               = emitter;

        if (ei.is_area()) {
            const ShapePtr shape = emitter->shape();
            PositionSample3f ps  = shape->sample_position(si.time, sampler->next_2d());
            if (ps.pdf > 0) {
                SurfaceInteraction3f si_emitter;
                si_emitter.p           = ps.p;
                si_emitter.wi          = Vector3f(0.f, 0.f, 1.f);
                si_emitter.wavelengths = si.wavelengths;
                si_emitter.time        = si.time;

                spec = emitter->eval(si_emitter) / ps.pdf;

                ei.p   = ps.p;
                ei.n   = ps.n;
                ei.d   = normalize(ps.p - si.p);
                ei.pdf = ps.pdf;
                ei.uv  = ps.uv;
            }
        } else if (ei.is_directional()) {
            auto [ds, spec_] = emitter->sample_direction(si, sampler->next_2d());
            ei.p             = ds.p;
            ei.d             = ds.d;
            ei.n             = ds.d;
            ei.pdf           = ds.pdf;
            spec             = spec_;
        } else if (ei.is_point()) {
            auto [ds, spec_] = emitter->sample_direction(si, sampler->next_2d());
            ei.p             = ds.p;
            ei.d             = ds.d;
            ei.n             = ei.d;
            ei.pdf           = ds.pdf;
            // Remove solid angle conversion factor. This will be accounted for
            // later in the geometric term computation.
            spec = spec_ * ds.dist * ds.dist;
        }

        ei.pdf *= emitter_pdf;
        ei.weight = spec * rcp(emitter_pdf);

        return ei;
    }

    /// Prepare emitter interaction for generalized geometry term computation
    static std::pair<Mask, ManifoldVertex> emitter_interaction_to_vertex(const Scene *scene,
                                                                         const EmitterInteraction &ei, const Point3f &p,
                                                                         Float time, const Wavelength &wavelengths,
                                                                         bool orthonomal = true) {
        if (ei.is_area()) {
            // Area emitters
            Vector3f d_tmp = normalize(ei.p - p);
            Ray3f ray_tmp(p + math::ShadowEpsilon<Float> * d_tmp, d_tmp, time, wavelengths);
            SurfaceInteraction3f si_y = scene->ray_intersect(ray_tmp);
            if (!si_y.is_valid()) {
                return std::make_pair(false, ManifoldVertex(Point3f(0.f)));
            }
            ManifoldVertex vy(si_y);
            if (orthonomal)
                vy.make_orthonormal();
            return std::make_pair(true, vy);
        } else if (ei.is_directional()) {
            // Directional & infinite emitters
            ManifoldVertex vy(ei.p);
            Vector3f d         = normalize(ei.p - p);
            vy.p               = p + d; // Place fake vertex at distance 1
            vy.n               = -d;
            auto [s, t]        = coordinate_system(vy.n);
            vy.dp_du           = s;
            vy.dp_dv           = t;
            vy.fixed_direction = true;
            return std::make_pair(true, vy);
        } else if (ei.is_point()) {
            // Point emitters
            ManifoldVertex vy(ei.p);
            Vector3f d = normalize(p - ei.p);
            vy.n = vy.gn = d;
            auto [s, t]  = coordinate_system(d);
            vy.dp_du     = s;
            vy.dp_dv     = t;
            return std::make_pair(true, vy);
        }
        return std::make_pair(false, ManifoldVertex(Point3f(0.f)));
    }

    /// Convert SurfaceInteraction into a EmitterInteraction struct
    static EmitterInteraction emitter_interaction(const Scene *scene, const SurfaceInteraction3f &si,
                                                  const SurfaceInteraction3f &si_emitter) {
        EmitterInteraction ei;

        const EmitterPtr emitter = si_emitter.emitter(scene);
        ei.emitter               = emitter;

        // Is either area light or infinite light, as it needs to be hit
        // explicitly in the scene.
        const ShapePtr shape = emitter->shape();
        if (shape) {
            ei.p = si_emitter.p;
            ei.n = si_emitter.n;
            ei.d = normalize(si_emitter.p - si.p);
        } else if (emitter->is_environment()) {
            ei.d = -si_emitter.wi;
            ei.p = si.p + 1000.f * ei.d;
            ei.n = si_emitter.wi;
        }
        return ei;
    }

    /// Sample the bivariate normal distribution for given mean vector and
    /// covariance matrix
    static MTS_INLINE Point2f sample_gaussian(const Point2f &mu, const Matrix2f &sigma, const Point2f &sample) {
        // Based on
        // https://math.stackexchange.com/questions/268298/sampling-from-a-2d-normal-with-a-given-covariance-matrix
        Point2f p     = warp::square_to_std_normal(sample);
        Float sigma_x = sqrt(sigma(0, 0)), sigma_y = sqrt(sigma(1, 1));

        Float rho = sigma(1, 0) / (sigma_x * sigma_y);
        Matrix2f P(0.5f, 0.5f, 0.5f, 0.5f);
        Matrix2f Q(0.5f, -0.5f, -0.5f, 0.5f);
        Matrix2f A = sqrt(1.f + rho) * P + sqrt(1.f - rho) * Q;
        p          = A * p;

        p[0] *= sigma_x;
        p[1] *= sigma_y;
        return p + mu;
    }

    static MTS_INLINE std::pair<Mask, Vector3f> reflect(const Vector3f &w, const Normal3f &n) {
        return std::make_pair(true, 2.f * dot(w, n) * n - w);
    }

    static MTS_INLINE std::pair<Vector3f, Vector3f> d_reflect(const Vector3f &w, const Vector3f &dw_du,
                                                              const Vector3f &dw_dv, const Normal3f &n,
                                                              const Vector3f &dn_du, const Vector3f &dn_dv) {
        Float dot_w_n = dot(w, n), dot_dwdu_n = dot(dw_du, n), dot_dwdv_n = dot(dw_dv, n), dot_w_dndu = dot(w, dn_du),
              dot_w_dndv = dot(w, dn_dv);
        Vector3f dwr_du  = 2.f * ((dot_dwdu_n + dot_w_dndu) * n + dot_w_n * dn_du) - dw_du,
                 dwr_dv  = 2.f * ((dot_dwdv_n + dot_w_dndv) * n + dot_w_n * dn_dv) - dw_dv;
        return std::make_pair(dwr_du, dwr_dv);
    }

    static MTS_INLINE std::pair<Mask, Vector3f> refract(const Vector3f &w, const Normal3f &n_, Float eta_) {
        Normal3f n = n_;
        Float eta  = rcp(eta_);
        if (dot(w, n) < 0) {
            // Coming from the "inside"
            eta = rcp(eta);
            n *= -1.f;
        }
        Float dot_w_n   = dot(w, n);
        Float root_term = 1.f - eta * eta * (1.f - dot_w_n * dot_w_n);
        if (root_term < 0.f) {
            return std::make_pair(false, Vector3f(0.f));
        }
        Vector3f wt = -eta * (w - dot_w_n * n) - n * sqrt(root_term);
        return std::make_pair(true, wt);
    }

    static MTS_INLINE std::pair<Vector3f, Vector3f> d_refract(const Vector3f &w, const Vector3f &dw_du,
                                                              const Vector3f &dw_dv, const Normal3f &n_,
                                                              const Vector3f &dn_du_, const Vector3f &dn_dv_,
                                                              Float eta_) {
        Normal3f n     = n_;
        Vector3f dn_du = dn_du_, dn_dv = dn_dv_;
        Float eta = rcp(eta_);
        if (dot(w, n) < 0) {
            // Coming from the "inside"
            eta = rcp(eta);
            n *= -1.f;
            dn_du *= -1.f;
            dn_dv *= -1.f;
        }
        Float dot_w_n = dot(w, n), dot_dwdu_n = dot(dw_du, n), dot_dwdv_n = dot(dw_dv, n), dot_w_dndu = dot(w, dn_du),
              dot_w_dndv = dot(w, dn_dv);
        Float root       = sqrt(1.f - eta * eta * (1.f - dot_w_n * dot_w_n));

        Vector3f a_u = -eta * (dw_du - ((dot_dwdu_n + dot_w_dndu) * n + dot_w_n * dn_du)), b1_u = dn_du * root,
                 b2_u = n * rcp(2.f * root) * (-eta * eta * (-2.f * dot_w_n * (dot_dwdu_n + dot_w_dndu))),
                 b_u = -(b1_u + b2_u), a_v = -eta * (dw_dv - ((dot_dwdv_n + dot_w_dndv) * n + dot_w_n * dn_dv)),
                 b1_v = dn_dv * root,
                 b2_v = n * rcp(2.f * root) * (-eta * eta * (-2.f * dot_w_n * (dot_dwdv_n + dot_w_dndv))),
                 b_v  = -(b1_v + b2_v);

        Vector3f dwt_du = a_u + b_u, dwt_dv = a_v + b_v;
        return std::make_pair(dwt_du, dwt_dv);
    }

    static MTS_INLINE std::pair<Float, Float> sphcoords(const Vector3f &w) {
        Float theta = safe_acos(w[2]);
        Float phi   = atan2(w[1], w[0]);
        if (phi < 0.f) {
            phi += 2.f * math::Pi<Float>;
        }
        return std::make_pair(theta, phi);
    }

    static MTS_INLINE std::tuple<Float, Float, Float, Float> d_sphcoords(const Vector3f &w, const Vector3f &dw_du,
                                                                         const Vector3f &dw_dv) {
        Float d_acos     = -rcp(safe_sqrt(1.f - w[2] * w[2]));
        Vector2f d_theta = d_acos * Vector2f(dw_du[2], dw_dv[2]);

        Float yx     = w[1] / w[0];
        Float d_atan = rcp(1 + yx * yx);
        Vector2f d_phi =
            d_atan * Vector2f(w[0] * dw_du[1] - w[1] * dw_du[0], w[0] * dw_dv[1] - w[1] * dw_dv[0]) * rcp(w[0] * w[0]);
        if (w[0] == 0.f) {
            d_phi = 0.f;
        }

        return std::make_tuple(d_theta[0], d_phi[0], d_theta[1], d_phi[1]);
    }

    /// Evalaute reflectance at specular interaction towards light source
    static Spectrum specular_reflectance(const Scene *scene, const SurfaceInteraction3f &si_,
                                         const EmitterInteraction &ei, const std::vector<ManifoldVertex> &v) {
        if (v.size() == 0)
            return 0.f;

        SurfaceInteraction3f si(si_);

        // Start ray-tracing towards the first specular vertex along the chain
        Point3f x0 = si.p, x1 = v[0].p;
        Vector3f wo = normalize(x1 - x0);
        Ray3f ray(x0, wo, si.time, si.wavelengths);

        Spectrum bsdf_val(1.f);

        for (size_t k = 0; k < v.size(); ++k) {
            si = scene->ray_intersect(ray);
            if (!si.is_valid()) {
                return 0.f;
            }

            // Prepare fore BSDF evaluation
            const BSDFPtr bsdf = si.bsdf();

            Vector3f wo;
            if (k < v.size() - 1) {
                // Safely connect with next vertex
                Point3f distribution_next = v[k + 1].p;
                wo                        = normalize(distribution_next - si.p);
            } else {
                // Connect with light source
                if (ei.is_directional()) {
                    wo = ei.d;
                } else {
                    wo = normalize(ei.p - si.p);
                }
            }

            Complex<Spectrum> ior = si.bsdf()->ior(si);
            Mask reflection       = any(neq(0.f, imag(ior)));
            reflection            = v[k].is_refract == false;

            Spectrum f;
            if (bsdf->roughness() > 0.f) {
                // Glossy BSDF: evaluate BSDF and transform to half-vector
                // domain.
                BSDFContext ctx;
                Vector3f wo_l = si.to_local(wo);
                f             = bsdf->eval(ctx, si, wo_l);

                /* Compared to Eq. 6 in [Hanika et al. 2015 (MNEE)], two terms
                are omitted: 1) abs_dot(wo, n) is part of BSDF::eval 2)
                abs_dot(h, n)  is part of the Microfacet distr. (also in
                BSDF::eval) */
                Vector3f h_l;
                if (reflection) {
                    h_l = normalize(si.wi + wo_l);
                    f *= 4.f * abs_dot(wo_l, h_l);
                } else {
                    Float eta = hmean(real(ior));
                    if (Frame3f::cos_theta(si.wi) < 0.f) {
                        eta = rcp(eta);
                    }
                    h_l = -normalize(si.wi + eta * wo_l);
                    f *= sqr(dot(si.wi, h_l) + eta * dot(wo_l, h_l)) / (eta * eta * abs_dot(wo_l, h_l));
                }

                bsdf_val *= f;
            } else {
                // Delta BSDF: just account for Fresnel term and solid angle
                // compression
                Frame3f frame   = bsdf->frame(si, 0.f);
                Float cos_theta = dot(frame.n, wo);
                if (reflection) {
                    if (all(eq(imag(ior), 0.f))) {
                        auto [F_, cos_theta_t, eta_it, eta_ti] = fresnel(Spectrum(abs(cos_theta)), real(ior));
                        f                                      = F_;
                    } else {
                        f = fresnel_conductor(Spectrum(abs(cos_theta)), ior);
                    }
                } else {
                    Float eta = hmean(real(ior));
                    if (cos_theta < 0.f) {
                        eta = rcp(eta);
                    }
                    auto [F, unused_0, unused_1, unused_2] = fresnel(cos_theta, eta);
                    f                                      = 1.f - F;
                    f *= sqr(eta);
                }

                bsdf_val *= f;
            }

            ray = Ray3f(si.p, wo, si.time, si.wavelengths);
        }

        // Do one more ray-trace towards light source to check for visibility
        // there
        if (!ei.is_directional()) {
            ray = Ray3f(si.p, normalize(ei.p - si.p), math::RayEpsilon<Float> * (1.f + hmax(abs(si.p))),
                        norm(ei.p - si.p) * (1.f - math::RayEpsilon<Float>), si.time, si.wavelengths);
        }

        if (scene->ray_test(ray)) {
            return 0.f;
        }

        return bsdf_val;
    }

    /// Compute generlized geometric term between vx and vy, via multiple
    /// specular vertices
    static Float geometric_term(const ManifoldVertex &vx, const ManifoldVertex &vy,
                                std::vector<ManifoldVertex> &current_path) {
        // First assemble full path, including all endpoints (use
        // m_proposed_path as buffer here)
        // m_proposed_path.clear();

        std::vector<ManifoldVertex> v;
        if (vy.fixed_direction) {
            // In this case, the resulting linear system is easier to solve if
            // we "reverse" the path actually
            v.push_back(vy);
            for (int i = current_path.size() - 1; i >= 0; --i) {
                v.push_back(current_path[i]);
            }
            v.push_back(vx);
        } else {
            // Assemble path in normal "forward" ordering, leaving out start
            // point 'x' as it's not needed
            for (size_t i = 0; i < current_path.size(); ++i) {
                v.push_back(current_path[i]);
            }
            v.push_back(vy);
        }

        // Do all the following work on this new path
        // std::vector<ManifoldVertex> &v = m_proposed_path;

        size_t k = v.size();
        for (size_t i = 0; i < k - 1; ++i) {
            v[i].dC_dx_prev = Matrix2f(0.f);
            v[i].dC_dx_cur  = Matrix2f(0.f);
            v[i].dC_dx_next = Matrix2f(0.f);

            Point3f x_cur  = v[i].p;
            Point3f x_next = v[i + 1].p;

            Vector3f wo = x_next - x_cur;
            Float ilo   = norm(wo);
            if (ilo < 1e-3f) {
                return false;
            }
            ilo = rcp(ilo);
            wo *= ilo;

            if (v[i].fixed_direction) {
                // Derivative of directional constraint w.r.t. x_{i}
                Vector3f dc_du_cur = -ilo * (v[i].dp_du - wo * dot(wo, v[i].dp_du)),
                         dc_dv_cur = -ilo * (v[i].dp_dv - wo * dot(wo, v[i].dp_dv));
                v[i].dC_dx_cur     = Matrix2f(dot(dc_du_cur, v[i].dp_du), dot(dc_dv_cur, v[i].dp_du),
                                              dot(dc_du_cur, v[i].dp_dv), dot(dc_dv_cur, v[i].dp_dv));

                // Derivative of directional constraint w.r.t. x_{i+1}
                Vector3f dc_du_next = ilo * (v[i + 1].dp_du - wo * dot(wo, v[i + 1].dp_du)),
                         dc_dv_next = ilo * (v[i + 1].dp_dv - wo * dot(wo, v[i + 1].dp_dv));
                v[i].dC_dx_next     = Matrix2f(dot(dc_du_next, v[i].dp_du), dot(dc_dv_next, v[i].dp_du),
                                               dot(dc_du_next, v[i].dp_dv), dot(dc_dv_next, v[i].dp_dv));
                continue;
            }

            Point3f x_prev = (i == 0) ? vx.p : v[i - 1].p; // Note that we only end up here
                                                           // for positionally fixed
                                                           // endpoints, thus x is not part
                                                           // of the path array directly.

            Vector3f wi = x_prev - x_cur;
            Float ili   = norm(wi);
            if (ili < 1e-3f) {
                return false;
            }
            ili = rcp(ili);
            wi *= ili;

            // Setup generalized half-vector
            Float eta = v[i].eta;
            if (v[i].is_refract == false) {
                eta = 1.f;
            }

            if (dot(wi, v[i].gn) < 0.f) {
                eta = rcp(eta);
            }
            Vector3f h = wi + eta * wo;
            if (eta != 1.f)
                h *= -1.f;
            Float ilh = rcp(norm(h));
            h *= ilh;

            ilo *= eta * ilh;
            ili *= ilh;

            // Local shading tangent frame
            Float dot_dpdu_n = dot(v[i].dp_du, v[i].n), dot_dpdv_n = dot(v[i].dp_dv, v[i].n);
            Vector3f s = v[i].dp_du - dot_dpdu_n * v[i].n, t = v[i].dp_dv - dot_dpdv_n * v[i].n;

            Vector3f dh_du, dh_dv;

            // Derivative of specular constraint w.r.t. x_{i-1}
            if (i > 0) {
                dh_du = ili * (v[i - 1].dp_du - wi * dot(wi, v[i - 1].dp_du));
                dh_dv = ili * (v[i - 1].dp_dv - wi * dot(wi, v[i - 1].dp_dv));
                dh_du -= h * dot(dh_du, h);
                dh_dv -= h * dot(dh_dv, h);
                if (eta != 1.f) {
                    dh_du *= -1.f;
                    dh_dv *= -1.f;
                }

                v[i].dC_dx_prev = Matrix2f(dot(s, dh_du), dot(s, dh_dv), dot(t, dh_du), dot(t, dh_dv));
            }

            // Derivative of specular constraint w.r.t. x_{i}
            dh_du = -v[i].dp_du * (ili + ilo) + wi * (dot(wi, v[i].dp_du) * ili) + wo * (dot(wo, v[i].dp_du) * ilo);
            dh_dv = -v[i].dp_dv * (ili + ilo) + wi * (dot(wi, v[i].dp_dv) * ili) + wo * (dot(wo, v[i].dp_dv) * ilo);
            dh_du -= h * dot(dh_du, h);
            dh_dv -= h * dot(dh_dv, h);
            if (eta != 1.f) {
                dh_du *= -1.f;
                dh_dv *= -1.f;
            }

            Float dot_h_n = dot(h, v[i].n), dot_h_dndu = dot(h, v[i].dn_du), dot_h_dndv = dot(h, v[i].dn_dv);

            v[i].dC_dx_cur = Matrix2f(dot(dh_du, s) - dot(v[i].dp_du, v[i].dn_du) * dot_h_n - dot_dpdu_n * dot_h_dndu,
                                      dot(dh_dv, s) - dot(v[i].dp_du, v[i].dn_dv) * dot_h_n - dot_dpdu_n * dot_h_dndv,
                                      dot(dh_du, t) - dot(v[i].dp_dv, v[i].dn_du) * dot_h_n - dot_dpdv_n * dot_h_dndu,
                                      dot(dh_dv, t) - dot(v[i].dp_dv, v[i].dn_dv) * dot_h_n - dot_dpdv_n * dot_h_dndv);

            // Derivative of specular constraint w.r.t. x_{i+1}
            dh_du = ilo * (v[i + 1].dp_du - wo * dot(wo, v[i + 1].dp_du));
            dh_dv = ilo * (v[i + 1].dp_dv - wo * dot(wo, v[i + 1].dp_dv));
            dh_du -= h * dot(dh_du, h);
            dh_dv -= h * dot(dh_dv, h);
            if (eta != 1.f) {
                dh_du *= -1.f;
                dh_dv *= -1.f;
            }

            v[i].dC_dx_next = Matrix2f(dot(s, dh_du), dot(s, dh_dv), dot(t, dh_du), dot(t, dh_dv));
        }

        if (vy.fixed_direction) {
            Float G = invert_tridiagonal_geo(v);
            // Cancel out cosine term that will be added during lighting
            // integral in integrator
            Vector3f d = normalize(v[k - 1].p - v[k - 2].p);
            G /= abs_dot(d, v[k - 1].n);
            return G;
        } else {
            Float dx1_dxend = invert_tridiagonal_geo(v);
            /* Unfortunately, these geometric terms can be unstable, so to avoid
            severe variance we can clamp here. */
            dx1_dxend    = min(dx1_dxend, Float(10.f));
            Vector3f d   = vx.p - v[0].p;
            Float inv_r2 = rcp(squared_norm(d));
            d *= sqrt(inv_r2);
            Float dw0_dx1 = abs_dot(d, v[0].gn) * inv_r2;
            Float G       = dw0_dx1 * dx1_dxend;
            return G;
        }
    }

    /// From the built-up tridiagonal block matrix, compute the geometric term
    static Float invert_tridiagonal_geo(std::vector<ManifoldVertex> &v) {
        auto invert = [](const Matrix2f &A, Matrix2f &Ainv) {
            Float determinant = det(A);
            if (abs(determinant) == 0) {
                return false;
            }
            Ainv = inverse(A);
            return true;
        };

        int n = int(v.size());
        if (n == 0)
            return 0.f;

        Matrix2f Li;
        if (!invert(v[0].dC_dx_cur, Li))
            return 0.f;

        for (int i = 0; i < n - 2; ++i) {
            v[i].tmp   = Li * v[i].dC_dx_next;
            Matrix2f m = v[i + 1].dC_dx_cur - v[i + 1].dC_dx_prev * v[i].tmp;
            if (!invert(m, Li))
                return 0.f;
        }

        v[n - 2].inv_lambda = -Li * v[n - 2].dC_dx_next;
        for (int i = n - 3; i >= 0; --i) {
            v[i].inv_lambda = -v[i].tmp * v[i + 1].inv_lambda;
        }
        return abs(det(-v[0].inv_lambda));
    }

    static Spectrum evaluate_path_contribution(const Scene *scene, const std::vector<ManifoldVertex> &current_path,
                                               SurfaceInteraction3f si, EmitterInteraction ei) {
        std::vector<ManifoldVertex> v = current_path;
        ManifoldVertex &vtx_last      = v[v.size() - 1];

        // Emitter to ManifoldVertex
        EmitterInteraction ei_(ei);
        if (ei_.is_point()) {
            /* Due to limitations in the API (e.g. no way to evaluate discrete
            emitter distributions directly), we need to re-sample this class of
            emitters one more time, in case they have some non-uniform emission
            profile (e.g. spotlights). It all works out because we're guaranteed
            to sample the same (delta) position again though and we don't even
            need a random sample. */
            SurfaceInteraction3f si_last(si);
            si_last.p          = vtx_last.p;
            si_last.n          = vtx_last.gn;
            si_last.sh_frame.n = vtx_last.n;
            auto [ds, spec]    = ei.emitter->sample_direction(si_last, Point2f(0.f));
            ei_.p              = ds.p;
            ei_.d              = ds.d;
            ei_.n              = ei.d;
            /* Because this is a delta light, ei.pdf just stores the discrete
            prob. of picking the initial emitter. It's important that we don't
            lose this quantity here when re-calculating the intensity. */
            Float emitter_pdf = ei_.pdf;
            ei_.pdf           = ds.pdf;
            // Remove solid angle conversion factor. This is accounted for in
            // the geometric term.
            ei_.weight = spec * ds.dist * ds.dist;

            ei_.pdf *= emitter_pdf;
            ei_.weight *= rcp(emitter_pdf);
        }
        auto [sucess_e, vy] =
            SpecularManifold::emitter_interaction_to_vertex(scene, ei_, vtx_last.p, si.time, si.wavelengths);
        if (!sucess_e) {
            return 0.f;
        }

        for (size_t k = 0; k < v.size(); ++k) {
            v[k].make_orthonormal();
        }

        // Shading point to ManifoldVertex
        ManifoldVertex vx(si, 0.f);
        vx.make_orthonormal();

        Spectrum path_throughput(1.f);
        path_throughput *= specular_reflectance(scene, si, ei_, v);
        path_throughput *= geometric_term(vx, vy, v);
        path_throughput *= ei_.weight;

        return path_throughput;
    }
};

template <typename Float, typename Spectrum> struct Manifold_Walk {
    MTS_IMPORT_TYPES(Sampler, Scene, Emitter)
    using EmitterPtr         = typename RenderAliases::EmitterPtr;
    using ShapePtr           = typename RenderAliases::ShapePtr;
    using ManifoldVertex     = ManifoldVertex<Float, Spectrum>;
    using EmitterInteraction = EmitterInteraction<Float, Spectrum>;
    /// Newton solver to find admissable path segment
    using SpecularManifold = SpecularManifold<Float, Spectrum>;

    Manifold_Walk() {}
    Manifold_Walk(const Scene *scene, ManifoldPathGuidingConfig config) : m_scene(scene), m_config(config) {}
    Mask newton_solver(const SurfaceInteraction3f &si, const EmitterInteraction &ei,
                       std::vector<ManifoldVertex> &current_path, const std::vector<Vector3f> &offset_normals) {
        // Newton iterations..
        bool success           = false;
        size_t iterations      = 0;
        Float beta             = 1.f;
        bool needs_step_update = true;
        std::vector<Point3f> proposed_positions;

        while (iterations < m_config.max_iterations) {
            bool step_success = true;
            if (needs_step_update) {
                if (m_config.halfvector_constraints) {
                    // Use standard manifold formulation using half-vector
                    // constraints
                    step_success = compute_step_halfvector(si, ei, current_path, offset_normals);
                } else {
                    // Use angle-difference constraint formulation
                    step_success = compute_step_anglediff(si, ei, current_path, offset_normals);
                }
            }
            if (!step_success) {
                break;
            }

            // Check for success
            bool converged = true;
            for (size_t i = 0; i < current_path.size(); ++i) {
                const ManifoldVertex &v = current_path[i];
                if (norm(v.C) > m_config.solver_threshold) {
                    converged = false;
                    break;
                }
            }
            if (converged) {
                success = true;
                break;
            }

            // Make a proposal
            proposed_positions.clear();
            for (size_t i = 0; i < current_path.size(); ++i) {
                const ManifoldVertex &v = current_path[i];
                Point3f p_prop          = v.p - m_config.step_scale * beta * (v.dp_du * v.dx[0] + v.dp_dv * v.dx[1]);
                proposed_positions.push_back(p_prop);
            }

            // Project back to surfaces
            auto [project_success, proposed_path] = reproject(si, proposed_positions, current_path, offset_normals);
            if (!project_success) {
                beta              = 0.5f * beta;
                needs_step_update = false;
            } else {
                beta              = std::min(Float(1), Float(2) * beta);
                current_path      = proposed_path;
                needs_step_update = true;
            }

            iterations++;
        }

        if (!success) {
            return false;
        }

        // m_newton_iters = iterations;

        /* In the refraction case, the half-vector formulation of Manifold
        walks will often converge to invalid solutions that are actually
        reflections. Here we need to reject those. */
        size_t n = current_path.size();
        for (size_t i = 0; i < n; ++i) {
            Point3f x_prev = (i == 0) ? si.p : current_path[i - 1].p;
            Point3f x_next = (i == n - 1) ? ei.p : current_path[i + 1].p;
            Point3f x_cur  = current_path[i].p;

            bool at_endpoint_with_fixed_direction = (i == (n - 1) && ei.is_directional());
            Vector3f wi                           = normalize(x_prev - x_cur);
            Vector3f wo                           = at_endpoint_with_fixed_direction ? ei.d : normalize(x_next - x_cur);

            Float cos_theta_i = dot(current_path[i].gn, wi), cos_theta_o = dot(current_path[i].gn, wo);
            bool refraction = cos_theta_i * cos_theta_o < 0.f, reflection = !refraction;
            if ((current_path[i].eta == 1.f && !reflection) || (current_path[i].eta != 1.f && !refraction)) {
                return false;
            }
        }

        return true;
    }

    /// Evaluate constraint and tangent step in the half-vector formulation
    Mask compute_step_halfvector(const SurfaceInteraction3f &si, const EmitterInteraction &ei,
                                 std::vector<ManifoldVertex> &v, const std::vector<Vector3f> offset_normals) {
        size_t k = v.size();
        for (size_t i = 0; i < k; ++i) {
            v[i].C          = Vector2f(0.f);
            v[i].dC_dx_prev = Matrix2f(0.f);
            v[i].dC_dx_cur  = Matrix2f(0.f);
            v[i].dC_dx_next = Matrix2f(0.f);

            Point3f x_prev = (i == 0) ? si.p : v[i - 1].p;
            Point3f x_next = (i == k - 1) ? ei.p : v[i + 1].p;
            Point3f x_cur  = v[i].p;

            bool at_endpoint_with_fixed_direction = (i == (k - 1) && ei.is_directional());

            // Setup wi / wo
            Vector3f wo;
            if (at_endpoint_with_fixed_direction) {
                // Case of fixed 'wo' direction
                wo = ei.d;
            } else {
                // Standard case for fixed emitter position
                wo = x_next - x_cur;
            }
            Float ilo = norm(wo);
            if (ilo < 1e-3f) {
                return false;
            }
            ilo = rcp(ilo);
            wo *= ilo;

            Vector3f wi = x_prev - x_cur;
            Float ili   = norm(wi);
            if (ili < 1e-3f) {
                return false;
            }
            ili = rcp(ili);
            wi *= ili;

            // Setup generalized half-vector
            Float eta = v[i].eta;
            if (dot(wi, v[i].gn) < 0.f) {
                eta = rcp(eta);
            }
            Vector3f h = wi + eta * wo;
            if (eta != 1.f)
                h *= -1.f;
            Float ilh = rcp(norm(h));
            h *= ilh;

            ilo *= eta * ilh;
            ili *= ilh;

            Vector3f dh_du, dh_dv;

            // Derivative of specular constraint w.r.t. x_{i-1}
            // if (i > 0) {
            if (i > 0) {
                dh_du = ili * (v[i - 1].dp_du - wi * dot(wi, v[i - 1].dp_du));
                dh_dv = ili * (v[i - 1].dp_dv - wi * dot(wi, v[i - 1].dp_dv));
            } else {
                dh_du = ili * (si.dp_du - wi * dot(wi, si.dp_du));
                dh_dv = ili * (si.dp_dv - wi * dot(wi, si.dp_dv));
            }

            dh_du -= h * dot(dh_du, h);
            dh_dv -= h * dot(dh_dv, h);
            if (eta != 1.f) {
                dh_du *= -1.f;
                dh_dv *= -1.f;
            }

            v[i].dC_dx_prev = Matrix2f(dot(v[i].s, dh_du), dot(v[i].s, dh_dv), dot(v[i].t, dh_du), dot(v[i].t, dh_dv));
            //}

            // Derivative of specular constraint w.r.t. x_{i}
            if (at_endpoint_with_fixed_direction) {
                // When the 'wo' direction is fixed, the derivative here
                // simplifies.
                dh_du = ili * (-v[i].dp_du + wi * dot(wi, v[i].dp_du));
                dh_dv = ili * (-v[i].dp_dv + wi * dot(wi, v[i].dp_dv));
            } else {
                // Standard case for fixed emitter position
                dh_du = -v[i].dp_du * (ili + ilo) + wi * (dot(wi, v[i].dp_du) * ili) + wo * (dot(wo, v[i].dp_du) * ilo);
                dh_dv = -v[i].dp_dv * (ili + ilo) + wi * (dot(wi, v[i].dp_dv) * ili) + wo * (dot(wo, v[i].dp_dv) * ilo);
            }
            dh_du -= h * dot(dh_du, h);
            dh_dv -= h * dot(dh_dv, h);
            if (eta != 1.f) {
                dh_du *= -1.f;
                dh_dv *= -1.f;
            }

            v[i].dC_dx_cur = Matrix2f(dot(v[i].ds_du, h) + dot(v[i].s, dh_du), dot(v[i].ds_dv, h) + dot(v[i].s, dh_dv),
                                      dot(v[i].dt_du, h) + dot(v[i].t, dh_du), dot(v[i].dt_dv, h) + dot(v[i].t, dh_dv));

            // Derivative of specular constraint w.r.t. x_{i+1}
            if (i < k - 1) {
                dh_du = ilo * (v[i + 1].dp_du - wo * dot(wo, v[i + 1].dp_du));
                dh_dv = ilo * (v[i + 1].dp_dv - wo * dot(wo, v[i + 1].dp_dv));
            } else {
                auto [success_e, vy] = SpecularManifold::emitter_interaction_to_vertex(m_scene, ei, v[i].p, si.time,
                                                                                       si.wavelengths, false);
                if (success_e) {
                    dh_du = ilo * (vy.dp_du - wo * dot(wo, vy.dp_du));
                    dh_dv = ilo * (vy.dp_dv - wo * dot(wo, vy.dp_dv));
                }
            }
            dh_du -= h * dot(dh_du, h);
            dh_dv -= h * dot(dh_dv, h);

            if (eta != 1.f) {
                dh_du *= -1.f;
                dh_dv *= -1.f;
            }

            v[i].dC_dx_next = Matrix2f(dot(v[i].s, dh_du), dot(v[i].s, dh_dv), dot(v[i].t, dh_du), dot(v[i].t, dh_dv));
            //}

            // Evaluate specular constraint
            Vector2f H(dot(v[i].s, h), dot(v[i].t, h));
            Vector3f n_offset = offset_normals[i];
            Vector2f N(n_offset[0], n_offset[1]);
            v[i].C = H - N;
        }

        if (!invert_tridiagonal_step(v)) {
            return false;
        }

        return true;
    }

    /// Evaluate constraint and tangent step in the angle difference formulation
    Mask compute_step_anglediff(const SurfaceInteraction3f &si, const EmitterInteraction &ei,
                                std::vector<ManifoldVertex> &v, const std::vector<Vector3f> &offset_normals) {
        bool success = true;
        size_t k     = v.size();
        for (size_t i = 0; i < k; ++i) {
            v[i].C          = Vector2f(0.f);
            v[i].dC_dx_prev = Matrix2f(0.f);
            v[i].dC_dx_cur  = Matrix2f(0.f);
            v[i].dC_dx_next = Matrix2f(0.f);

            Point3f x_prev = (i == 0) ? si.p : v[i - 1].p;
            Point3f x_next = (i == k - 1) ? ei.p : v[i + 1].p;
            Point3f x_cur  = v[i].p;

            bool at_endpoint_with_fixed_direction = (i == (k - 1) && ei.is_directional());

            // Setup wi / wo
            Vector3f wo;
            if (at_endpoint_with_fixed_direction) {
                // Case of fixed 'wo' direction
                wo = ei.d;
            } else {
                // Standard case for fixed emitter position
                wo = x_next - x_cur;
            }
            Float ilo = norm(wo);
            if (ilo < 1e-3f) {
                return false;
            }
            ilo = rcp(ilo);
            wo *= ilo;

            Vector3f dwo_du_cur, dwo_dv_cur;
            if (at_endpoint_with_fixed_direction) {
                // Fixed 'wo' direction means its derivative must be zero
                dwo_du_cur = Vector3f(0.f);
                dwo_dv_cur = Vector3f(0.f);
            } else {
                // Standard case for fixed emitter position
                dwo_du_cur = -ilo * (v[i].dp_du - wo * dot(wo, v[i].dp_du));
                dwo_dv_cur = -ilo * (v[i].dp_dv - wo * dot(wo, v[i].dp_dv));
            }

            Vector3f wi = x_prev - x_cur;
            Float ili   = norm(wi);
            if (ili < 1e-3f) {
                return false;
            }
            ili = rcp(ili);
            wi *= ili;

            Vector3f dwi_du_cur = -ili * (v[i].dp_du - wi * dot(wi, v[i].dp_du)),
                     dwi_dv_cur = -ili * (v[i].dp_dv - wi * dot(wi, v[i].dp_dv));

            // Set up constraint function and its derivatives
            bool success_i = false;

            auto transform = [&](const Vector3f &w, const Vector3f &n, Float eta) {
                // if (eta == 1.f) {
                if (!v[i].is_refract) {
                    return SpecularManifold::reflect(w, n);
                } else {
                    return SpecularManifold::refract(w, n, eta);
                }
            };
            auto d_transform = [&](const Vector3f &w, const Vector3f &dw_du, const Vector3f &dw_dv, const Vector3f &n,
                                   const Vector3f &dn_du, const Vector3f &dn_dv, Float eta) {
                // if (eta == 1.f) {
                if (!v[i].is_refract) {
                    return SpecularManifold::d_reflect(w, dw_du, dw_dv, n, dn_du, dn_dv);
                } else {
                    return SpecularManifold::d_refract(w, dw_du, dw_dv, n, dn_du, dn_dv, eta);
                }
            };

            // Handle offset normal. These are no-ops in case n_offset=[0,0,1]
            Vector3f n_offset = offset_normals[i];
            Normal3f n        = v[i].s * n_offset[0] + v[i].t * n_offset[1] + v[i].n * n_offset[2];
            Vector3f dn_du    = v[i].ds_du * n_offset[0] + v[i].dt_du * n_offset[1] + v[i].dn_du * n_offset[2];
            Vector3f dn_dv    = v[i].ds_dv * n_offset[0] + v[i].dt_dv * n_offset[1] + v[i].dn_dv * n_offset[2];

            auto [valid_i_refr_i, wio] = transform(wi, n, v[i].eta);
            if (valid_i_refr_i) {
                auto [to, po]   = SpecularManifold::sphcoords(wo);
                auto [tio, pio] = SpecularManifold::sphcoords(wio);

                Float dt = to - tio, dp = po - pio;
                if (dp < -math::Pi<Float>) {
                    dp += 2.f * math::Pi<Float>;
                } else if (dp > math::Pi<Float>) {
                    dp -= 2.f * math::Pi<Float>;
                }
                v[i].C = Vector2f(dt, dp);

                Float dto_du, dpo_du, dto_dv, dpo_dv;
                Float dtio_du, dpio_du, dtio_dv, dpio_dv;

                // Derivative of specular constraint w.r.t. x_{i-1}
                Vector3f dwi_du_prev, dwi_dv_prev;
                if (i > 0) {
                    dwi_du_prev = ili * (v[i - 1].dp_du - wi * dot(wi, v[i - 1].dp_du)),
                    dwi_dv_prev = ili * (v[i - 1].dp_dv - wi * dot(wi, v[i - 1].dp_dv));
                } else {
                    dwi_du_prev = ili * (si.dp_du - wi * dot(wi, si.dp_du)),
                    dwi_dv_prev = ili * (si.dp_dv - wi * dot(wi, si.dp_dv));
                }
                // Vector3f dwo_du_prev = ilo * (v[i-1].dp_du - wo*dot(wo,
                // v[i-1].dp_du)),  // = 0
                //          dwo_dv_prev = ilo * (v[i-1].dp_dv - wo*dot(wo,
                //          v[i-1].dp_dv));  // = 0
                auto [dwio_du_prev, dwio_dv_prev] =
                    d_transform(wi, dwi_du_prev, dwi_dv_prev, n, Vector3f(0.f), Vector3f(0.f),
                                v[i].eta); // Possible optimization: specific implementation
                                           // here that already knows some of these are 0.

                // std::tie(dto_du, dpo_du, dto_dv, dpo_dv)     =
                // SpecularManifold::d_sphcoords(wo, dwo_du_prev, dwo_dv_prev);
                // // = 0
                std::tie(dtio_du, dpio_du, dtio_dv, dpio_dv) =
                    SpecularManifold::d_sphcoords(wio, dwio_du_prev, dwio_dv_prev);

                v[i].dC_dx_prev(0, 0) = -dtio_du;
                v[i].dC_dx_prev(1, 0) = -dpio_du;
                v[i].dC_dx_prev(0, 1) = -dtio_dv;
                v[i].dC_dx_prev(1, 1) = -dpio_dv;
                //}

                // Derivative of specular constraint w.r.t. x_{i}
                auto [dwio_du_cur, dwio_dv_cur] = d_transform(wi, dwi_du_cur, dwi_dv_cur, n, dn_du, dn_dv, v[i].eta);

                std::tie(dto_du, dpo_du, dto_dv, dpo_dv) = SpecularManifold::d_sphcoords(wo, dwo_du_cur, dwo_dv_cur);
                std::tie(dtio_du, dpio_du, dtio_dv, dpio_dv) =
                    SpecularManifold::d_sphcoords(wio, dwio_du_cur, dwio_dv_cur);

                v[i].dC_dx_cur(0, 0) = dto_du - dtio_du;
                v[i].dC_dx_cur(1, 0) = dpo_du - dpio_du;
                v[i].dC_dx_cur(0, 1) = dto_dv - dtio_dv;
                v[i].dC_dx_cur(1, 1) = dpo_dv - dpio_dv;

                // Derivative of specular constraint w.r.t. x_{i+1}
                Vector3f dwo_du_next, dwo_dv_next;
                if (i < k - 1) {
                    // Vector3f dwi_du_next = ili * (v[i+1].dp_du - wi*dot(wi,
                    // v[i+1].dp_du)),  // = 0 dwi_dv_next = ili * (v[i+1].dp_dv
                    // - wi*dot(wi, v[i+1].dp_dv));  // = 0
                    dwo_du_next = ilo * (v[i + 1].dp_du - wo * dot(wo, v[i + 1].dp_du)),
                    dwo_dv_next = ilo * (v[i + 1].dp_dv - wo * dot(wo, v[i + 1].dp_dv));
                } else {
                    auto [success_e, vy] = SpecularManifold::emitter_interaction_to_vertex(m_scene, ei, v[i].p, si.time,
                                                                                           si.wavelengths, false);
                    if (success_e) {
                        dwo_du_next = ilo * (vy.dp_du - wo * dot(wo, vy.dp_du)),
                        dwo_dv_next = ilo * (vy.dp_dv - wo * dot(wo, vy.dp_dv));
                    }
                }
                // auto [dwio_du_next, dwio_dv_next] = d_transform(wi,
                // dwi_du_next, dwi_dv_next, n, Vector3f(0.f), Vector3f(0.f),
                // v[i].eta); // = 0

                std::tie(dto_du, dpo_du, dto_dv, dpo_dv) = SpecularManifold::d_sphcoords(wo, dwo_du_next, dwo_dv_next);
                // std::tie(dtio_du, dpio_du, dtio_dv, dpio_dv) =
                // SpecularManifold::d_sphcoords(wio, dwio_du_next,
                // dwio_dv_next);   // = 0

                v[i].dC_dx_next(0, 0) = dto_du;
                v[i].dC_dx_next(1, 0) = dpo_du;
                v[i].dC_dx_next(0, 1) = dto_dv;
                v[i].dC_dx_next(1, 1) = dpo_dv;
                //}

                success_i = true;
            }

            auto [valid_o_refr_o, woi] = transform(wo, n, v[i].eta);
            if (valid_o_refr_o && !success_i) {
                auto [ti, pi]   = SpecularManifold::sphcoords(wi);
                auto [toi, poi] = SpecularManifold::sphcoords(woi);

                Float dt = ti - toi, dp = pi - poi;
                if (dp < -math::Pi<Float>) {
                    dp += 2.f * math::Pi<Float>;
                } else if (dp > math::Pi<Float>) {
                    dp -= 2.f * math::Pi<Float>;
                }
                v[i].C = Vector2f(dt, dp);

                Float dti_du, dpi_du, dti_dv, dpi_dv;
                Float dtoi_du, dpoi_du, dtoi_dv, dpoi_dv;

                // Derivative of specular constraint w.r.t. x_{i-1}
                Vector3f dwi_du_prev, dwi_dv_prev;
                if (i > 0) {
                    dwi_du_prev = ili * (v[i - 1].dp_du - wi * dot(wi, v[i - 1].dp_du)),
                    dwi_dv_prev = ili * (v[i - 1].dp_dv - wi * dot(wi, v[i - 1].dp_dv));
                } else {
                    dwi_du_prev = ili * (si.dp_du - wi * dot(wi, si.dp_du)),
                    dwi_dv_prev = ili * (si.dp_dv - wi * dot(wi, si.dp_dv));
                }
                // Vector3f dwo_du_prev = ilo * (v[i-1].dp_du - wo*dot(wo,
                // v[i-1].dp_du)),  // = 0 dwo_dv_prev = ilo * (v[i-1].dp_dv -
                // wo*dot(wo, v[i-1].dp_dv));  // = 0 auto [dwoi_du_prev,
                // dwoi_dv_prev] = d_transform(wo, dwo_du_prev, dwo_dv_prev, n,
                // Vector3f(0.f), Vector3f(0.f), v[i].eta);   // = 0

                std::tie(dti_du, dpi_du, dti_dv, dpi_dv) = SpecularManifold::d_sphcoords(wi, dwi_du_prev, dwi_dv_prev);
                // std::tie(dtoi_du, dpoi_du, dtoi_dv, dpoi_dv) =
                // SpecularManifold::d_sphcoords(woi, dwoi_du_prev,
                // dwoi_dv_prev);   // = 0

                v[i].dC_dx_prev(0, 0) = dti_du;
                v[i].dC_dx_prev(1, 0) = dpi_du;
                v[i].dC_dx_prev(0, 1) = dti_dv;
                v[i].dC_dx_prev(1, 1) = dpi_dv;
                //}

                // Derivative of specular constraint w.r.t. x_{i}
                auto [dwoi_du_cur, dwoi_dv_cur] = d_transform(wo, dwo_du_cur, dwo_dv_cur, n, dn_du, dn_dv, v[i].eta);

                std::tie(dti_du, dpi_du, dti_dv, dpi_dv) = SpecularManifold::d_sphcoords(wi, dwi_du_cur, dwi_dv_cur);
                std::tie(dtoi_du, dpoi_du, dtoi_dv, dpoi_dv) =
                    SpecularManifold::d_sphcoords(woi, dwoi_du_cur, dwoi_dv_cur);

                v[i].dC_dx_cur(0, 0) = dti_du - dtoi_du;
                v[i].dC_dx_cur(1, 0) = dpi_du - dpoi_du;
                v[i].dC_dx_cur(0, 1) = dti_dv - dtoi_dv;
                v[i].dC_dx_cur(1, 1) = dpi_dv - dpoi_dv;

                // Derivative of specular constraint w.r.t. x_{i+1}
                Vector3f dwo_du_next, dwo_dv_next;
                if (i < k - 1) {
                    // Vector3f dwi_du_next = ili * (v[i+1].dp_du - wi*dot(wi,
                    // v[i+1].dp_du)),  // = 0 dwi_dv_next = ili * (v[i+1].dp_dv
                    // - wi*dot(wi, v[i+1].dp_dv));  // = 0
                    dwo_du_next = ilo * (v[i + 1].dp_du - wo * dot(wo, v[i + 1].dp_du)),
                    dwo_dv_next = ilo * (v[i + 1].dp_dv - wo * dot(wo, v[i + 1].dp_dv));
                } else {
                    auto [success_e, vy] = SpecularManifold::emitter_interaction_to_vertex(m_scene, ei, v[i].p, si.time,
                                                                                           si.wavelengths, false);
                    if (success_e) {
                        dwo_du_next = ilo * (vy.dp_du - wo * dot(wo, vy.dp_du)),
                        dwo_dv_next = ilo * (vy.dp_dv - wo * dot(wo, vy.dp_dv));
                    }
                }
                auto [dwoi_du_next, dwoi_dv_next] =
                    d_transform(wo, dwo_du_next, dwo_dv_next, n, Vector3f(0.f), Vector3f(0.f),
                                v[i].eta); // Possible optimization: specific implementation
                                           // here that already knows some of these are 0.

                // std::tie(dti_du, dpi_du, dti_dv, dpi_dv)  =
                // SpecularManifold::d_sphcoords(wi, dwi_du_next, dwi_dv_next);
                // // = 0
                std::tie(dtoi_du, dpoi_du, dtoi_dv, dpoi_dv) =
                    SpecularManifold::d_sphcoords(woi, dwoi_du_next, dwoi_dv_next);

                v[i].dC_dx_next(0, 0) = -dtoi_du;
                v[i].dC_dx_next(1, 0) = -dpoi_du;
                v[i].dC_dx_next(0, 1) = -dtoi_dv;
                v[i].dC_dx_next(1, 1) = -dpoi_dv;
                //}

                success_i = true;
            }

            success &= success_i;
        }

        if (!success || !invert_tridiagonal_step(v)) {
            return false;
        }
        return true;
    }

    /// Reproject proposed offset positions back to surfaces
    std::pair<Mask, std::vector<ManifoldVertex>> reproject(const SurfaceInteraction3f &si_,
                                                           const std::vector<Point3f> &proposed_positions,
                                                           const std::vector<ManifoldVertex> &current_path,
                                                           const std::vector<Vector3f> offset_normals) {

        SurfaceInteraction3f si(si_);

        // Start ray-tracing towards the first specular vertex along the chain
        Point3f x0 = si.p, x1 = proposed_positions[0];
        Vector3f wo = normalize(x1 - x0);
        Ray3f ray(x0, wo, si.time, si.wavelengths);
        std::vector<ManifoldVertex> proposed_path;
        while (true) {
            int bounce = proposed_path.size();

            if (bounce >= current_path.size()) {
                break;
            }

            // if (bounce >= m_config.bounces) {
            //     /* We reached the number of specular bounces that was
            //     requested.
            //        (Implicitly) connect to the light source now by
            //        terminating. */
            //     break;
            // }

            si = m_scene->ray_intersect(ray);
            if (!si.is_valid()) {
                return std::make_pair(false, proposed_path);
            }
            const ShapePtr shape = si.shape;
            if (shape != current_path[bounce].shape) {
                // We hit some other shape than previously
                return std::make_pair(false, proposed_path);
            }

            // Create the path vertex
            ManifoldVertex vertex = ManifoldVertex(si, 0.f);
            vertex.is_refract     = current_path[bounce].is_refract;
            proposed_path.push_back(vertex);

            // Get current (potentially offset) normal in world space
            Vector3f n_offset = offset_normals[bounce];
            Vector3f m        = vertex.s * n_offset[0] + vertex.t * n_offset[1] + vertex.n * n_offset[2];

            // Perform scattering at vertex
            Vector3f wi          = -wo;
            bool scatter_success = false;
            if (!vertex.is_refract) {
                // if (vertex.eta == 1.f) {
                std::tie(scatter_success, wo) = SpecularManifold::reflect(wi, m);
            } else {
                std::tie(scatter_success, wo) = SpecularManifold::refract(wi, m, vertex.eta);
            }

            if (!scatter_success) {
                // We must have hit total internal reflection. Abort.
                return std::make_pair(false, proposed_path);
            }

            ray = si.spawn_ray(wo);
        }

        return std::make_pair(proposed_path.size() == current_path.size(), proposed_path);
    }

    /// From the built-up tridiagonal block matrix, compute the steps
    Mask invert_tridiagonal_step(std::vector<ManifoldVertex> &v) {

        auto invert = [](const Matrix2f &A, Matrix2f &Ainv) {
            Float determinant = det(A);
            if (abs(determinant) == 0) {
                return false;
            }
            Ainv = inverse(A);
            return true;
        };

        int n = int(v.size());
        if (n == 0)
            return true;

        v[0].tmp   = v[0].dC_dx_prev;
        Matrix2f m = v[0].dC_dx_cur;
        if (!invert(m, v[0].inv_lambda))
            return false;

        for (int i = 1; i < n; ++i) {
            v[i].tmp   = v[i].dC_dx_prev * v[i - 1].inv_lambda;
            Matrix2f m = v[i].dC_dx_cur - v[i].tmp * v[i - 1].dC_dx_next;
            if (!invert(m, v[i].inv_lambda))
                return false;
        }

        v[0].dx = v[0].C;
        for (int i = 1; i < n; ++i) {
            v[i].dx = v[i].C - v[i].tmp * v[i - 1].dx;
        }

        v[n - 1].dx = v[n - 1].inv_lambda * v[n - 1].dx;
        for (int i = n - 2; i >= 0; --i) {
            v[i].dx = v[i].inv_lambda * (v[i].dx - v[i].dC_dx_next * v[i + 1].dx);
        }
        return true;
    }
    ManifoldPathGuidingConfig m_config;
    const Scene *m_scene = nullptr;
};

NAMESPACE_END(mitsuba)
