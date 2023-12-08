#pragma once

#include "util.h"
#include "dtree.h"
#include "chain_distribution.h"

NAMESPACE_BEGIN(mitsuba)


////////////////////////////
// Spatial Structure
/////////////////////////////

template <typename Float, typename Spectrum> struct SpatialStructure {
    MTS_IMPORT_TYPES();
    using SubpathSample     = SubpathSample<Float, Spectrum>;
    using ChainDistribution = ChainDistribution<Float, Spectrum>;
    using AABB              = BoundingBox<Point3>;

public:
    SpatialStructure() {}
    virtual void build(const std::vector<SubpathSample> &subpath_sample_list, const float spatial_radius) = 0;
    virtual std::shared_ptr<ChainDistribution> query(const Point3f xD, const Point3f xL)                  = 0;
    virtual void print_stats()                                                                            = 0;
};

int g_iter = 0;



// ANN library
template <typename Float, typename Spectrum> struct SpatialStructureANN : public SpatialStructure<Float, Spectrum> {
    MTS_IMPORT_TYPES();
    using SpatialStructure = SpatialStructure<Float, Spectrum>;

public:
    SpatialStructureANN(AABB aabb) {}
    ~SpatialStructureANN() { delete kdtree; }

    virtual void build(const std::vector<SubpathSample> &subpath_sample_list, const float spatial_radius) {
        NanosecondTimer perf_timer;
        double min_samples = 32;

        // multiplexed parameter "spatial_radius"
        if (spatial_radius > 1e4) {
            knn_k = std::max(min_samples, spatial_radius * 1e-6 * std::pow(subpath_sample_list.size(), 0.5f));
        } else if (spatial_radius > 0) {
            knn_k = spatial_radius;
        } else if (spatial_radius >= -1) {
            knn_k = std::max(min_samples, std::pow(subpath_sample_list.size(), 0.5f));
        } else {
            knn_k = -spatial_radius * std::pow(2.0, g_iter * 0.5);
        }

        knn_data.clear();
        for (const auto &i : subpath_sample_list) {
            knn_data.push_back(i);
        }
        std::vector<std::vector<float>> vec_configurations;
        for (const auto &s : knn_data) {
            vec_configurations.push_back({ s.xD[0], s.xD[1], s.xD[2], s.xL[0], s.xL[1], s.xL[2] });
        }
        configurations = annAllocPts(vec_configurations.size(), 6);
        for (int j = 0; j < vec_configurations.size(); j++) {
            for (int k = 0; k < 6; k++) {
                configurations[j][k] = vec_configurations[j][k];
            }
        }
        delete kdtree;
        kdtree = new ANNkd_tree(configurations, vec_configurations.size(), 6);
        perf_time_build += (float) perf_timer.value();
    }

    std::shared_ptr<ChainDistribution> query(const Point3f xD, const Point3f xL) {
        NanosecondTimer timer;
        if (knn_data.size() == 0) {
            return {};
        }

        int k                  = std::min(knn_k, (int) knn_data.size());
        double q[6]            = { xD[0], xD[1], xD[2], xL[0], xL[1], xL[2] };
        ANNpoint query_pos     = q;
        int *ann_idx_array     = new int[k + 2];
        double *ann_dist_array = new double[k + 2];
        kdtree->annkSearch(query_pos, k, ann_idx_array, ann_dist_array, 0.0);

        std::vector<SubpathSample> knn_result;
        for (int i = 0; i < k; i++) {
            knn_result.push_back(knn_data[ann_idx_array[i]]);
        }

        std::shared_ptr<ChainDistribution> ans = chainDistributionFactory<Float, Spectrum>();
        ans->load_samples(knn_result);
        ans->build_distribution();

        delete[] ann_idx_array;
        delete[] ann_dist_array;

        perf_time_query += timer.value();
        perf_num_query++;

        return ans;
    }

    void print_stats() {
        std::cout << "KNN stats perf_time_build " << std::fixed << std::setprecision(6) << perf_time_build * 1e-9
                  << std::endl;
        std::cout << "KNN stats perf_time_query " << std::fixed << std::setprecision(6) << perf_time_query * 1e-9
                  << std::endl;
        std::cout << "KNN stats perf_num_query " << perf_num_query << std::endl;
        std::cout << "KNN stats perf_time_query_avg " << perf_time_query * 1e-9 / perf_num_query << std::endl;
    }

public:
    ANNpointArray configurations;
    ANNkd_tree *kdtree = nullptr;
    int knn_k;
    std::vector<SubpathSample> knn_data;

    static inline std::atomic<unsigned long long> perf_time_build = 0; // nano sec
    static inline std::atomic<unsigned long long> perf_time_query = 0; // nano sec
    static inline std::atomic<unsigned long long> perf_num_query  = 0; // n
};


// STree
template <typename Float, typename Spectrum> struct MySTreeNode {
    MTS_IMPORT_TYPES(BSDF, Sampler, Scene, Shape);

    using SubpathSample     = SubpathSample<Float, Spectrum>;
    using ChainDistribution = ChainDistribution<Float, Spectrum>;

    MySTreeNode() {
        children   = {};
        isLeaf     = true;
        axis       = 0;
        distr      = chainDistributionFactory<Float, Spectrum>();
        axis_alpha = 0.5;
    }

    int childIndex(Point6f &p) const {
        if (p[axis] < axis_alpha) {
            p[axis] /= axis_alpha;
            return 0;
        } else {
            p[axis] = (p[axis] - axis_alpha) / (1 - axis_alpha);
            return 1;
        }
    }

    int nodeIndex(Point6f &p) const { return children[childIndex(p)]; }

    std::shared_ptr<ChainDistribution> chainDistribution(Point6f &p, Vector6f &size, std::vector<MySTreeNode> &nodes) {
        if (isLeaf) {
            return distr;
        } else {
            size[axis] /= 2;
            return nodes[nodeIndex(p)].chainDistribution(p, size, nodes);
        }
    }

    std::shared_ptr<ChainDistribution> chainDistribution(Point6f &p, std::vector<MySTreeNode> &nodes) {
        if (isLeaf) {
            return distr;
        } else {
            return nodes[nodeIndex(p)].chainDistribution(p, nodes);
        }
    }

    const std::shared_ptr<ChainDistribution> chainDistribution() const { return distr; }

    int depth(Point3 &p, const std::vector<MySTreeNode> &nodes) const {
        if (isLeaf) {
            return 1;
        } else {
            return 1 + nodes[nodeIndex(p)].depth(p, nodes);
        }
    }

    int depth(const std::vector<MySTreeNode> &nodes) const {
        int result = 1;

        if (!isLeaf) {
            for (auto c : children) {
                result = std::max(result, 1 + nodes[c].depth(nodes));
            }
        }

        return result;
    }

    void
    forEachLeaf(std::function<void(std::shared_ptr<const ChainDistribution>, const Point3 &, const Vector3 &)> func,
                Point3 p, Vector3 size, const std::vector<MySTreeNode> &nodes) const {

        if (isLeaf) {
            func(&distr, p, size);
        } else {
            size[axis] /= 2;
            for (int i = 0; i < 2; ++i) {
                Point3 childP = p;
                if (i == 1) {
                    childP[axis] += size[axis];
                }

                nodes[children[i]].forEachLeaf(func, childP, size, nodes);
            }
        }
    }

    Float computeOverlappingVolume(const Point6f &min1, const Point6f &max1, const Point6f &min2, const Point6f &max2) {
        Float lengths[6];
        for (int i = 0; i < 6; ++i) {
            lengths[i] = std::max(std::min(max1[i], max2[i]) - std::max(min1[i], min2[i]), 0.0f);
        }
        return lengths[0] * lengths[1] * lengths[2] * lengths[3] * lengths[4] * lengths[5];
    }

    void record(const Point6f &min1, const Point6f &max1, Point6f min2, Vector6f size2, const SubpathSample &rec,
                std::vector<MySTreeNode> &nodes, Float original_volume, int &recordCnt) {
        Float w = computeOverlappingVolume(min1, max1, min2, min2 + size2) / original_volume;
        if (w > 0) {
            if (isLeaf) {
                recordCnt += 1;
                distr->record(rec, w);
            } else {
                size2[axis] /= 2;
                for (int i = 0; i < 2; ++i) {
                    if (i & 1) {
                        min2[axis] += size2[axis];
                    }
                    nodes[children[i]].record(min1, max1, min2, size2, rec, nodes, original_volume, recordCnt);
                }
            }
        }
    }

    void record(const Point6f &min1, const Point6f &max1, Point6f min2, Vector6f size2, const SubpathSample &rec,
                std::vector<MySTreeNode> &nodes, Float original_volume) {
        int rc = 0;
        record(min1, max1, min2, size2, rec, nodes, original_volume, rc);
    }

    bool isLeaf;
    std::shared_ptr<ChainDistribution> distr;
    int axis;
    float axis_alpha = 0.5;
    std::array<uint32_t, 2> children;
};

template <typename Float, typename Spectrum> class SpatialStructureSTree : public SpatialStructure<Float, Spectrum> {
public:
    MTS_IMPORT_TYPES(BSDF, Sampler, Scene, Shape);
    using MySTreeNode = MySTreeNode<Float, Spectrum>;

    SpatialStructureSTree(AABB aabb) {
        clear();
        m_aabb        = aabb;
        Vector3 size  = m_aabb.max - m_aabb.min;
        Float maxSize = std::max(std::max(size[0], size[1]), size[2]);
        m_aabb.max    = m_aabb.min + Vector3(maxSize);
    }

    void clear() {
        m_nodes.clear();
        m_nodes.emplace_back();
    }

    std::atomic<int> m_node_size = 1;

    int stree_nodes_limit = 1e5;

    void subdivide(int nodeIdx, std::vector<MySTreeNode> &nodes, Vector6f size, Point6f offset) {
        float eps = 0.1;
        int cur_node_num;
#pragma omp critical
        cur_node_num = std::atomic_fetch_add(&m_node_size, 2) + 2;

        if (cur_node_num >= stree_nodes_limit) {
            std::cout << "error   cur_node_num >= limit" << std::endl;
            return;
        }

        MySTreeNode &cur          = nodes[nodeIdx];
        auto &subpath_sample_list = cur.distr->m_subpath_sample_list;
        auto &aabb                = cur.distr->aabb;

        cur.axis_alpha    = 0.5;
        Float cur_left    = offset[cur.axis];
        Float cur_right   = offset[cur.axis] + size[cur.axis];
        Float split_value = offset[cur.axis] + size[cur.axis] * cur.axis_alpha;

        if (aabb.max[cur.axis] < cur_left + (cur_right - cur_left) * (1.0 / 2 - eps)) {
            int i            = 0;
            uint32_t idx     = (uint32_t) cur_node_num - 2 + i;
            cur.children[i]  = idx;
            nodes[idx].axis  = (cur.axis + 1) % 6;
            nodes[idx].distr = cur.distr;
            i                = 1;
            idx              = (uint32_t) cur_node_num - 2 + i;
            cur.children[i]  = idx;
            nodes[idx].axis  = (cur.axis + 1) % 6;
            nodes[idx].distr = chainDistributionFactory<Float, Spectrum>();
            nodes[idx].distr->load_samples({});
            cur.isLeaf = false;
            cur.distr  = nullptr;
        } else if (aabb.min[cur.axis] > cur_left + (cur_right - cur_left) * (1.0 / 2 + eps)) {
            int i            = 1;
            uint32_t idx     = (uint32_t) cur_node_num - 2 + i;
            cur.children[i]  = idx;
            nodes[idx].axis  = (cur.axis + 1) % 6;
            nodes[idx].distr = cur.distr;
            i                = 0;
            idx              = (uint32_t) cur_node_num - 2 + i;
            cur.children[i]  = idx;
            nodes[idx].axis  = (cur.axis + 1) % 6;
            nodes[idx].distr = chainDistributionFactory<Float, Spectrum>();
            nodes[idx].distr->load_samples({});
            cur.isLeaf = false;
            cur.distr  = nullptr;
            return;
        }

        std::vector<SubpathSample> children_samples[2];
        for (const auto &i : subpath_sample_list) {
            Float cur_value = cur.axis < 3 ? i.xD[cur.axis] : i.xL[cur.axis - 3];
            if (cur_value <= split_value)
                children_samples[0].push_back(i);
            else
                children_samples[1].push_back(i);
        }

        if (global_sms_config.spatial_filter > 0) {
            int overlap = std::min((int) subpath_sample_list.size(), stree_thres) * global_sms_config.spatial_filter;
            int mid     = children_samples[0].size();

            std::nth_element(subpath_sample_list.begin(), subpath_sample_list.begin() + mid, subpath_sample_list.end(),
                             [&](const SubpathSample &a, const SubpathSample &b) {
                                 if (cur.axis < 3)
                                     return a.xD[cur.axis] < b.xD[cur.axis];
                                 else
                                     return a.xL[cur.axis - 3] < b.xL[cur.axis - 3];
                             });

            int left_bound  = std::max(0, mid - overlap);
            int right_bound = std::max(1, std::min((int) subpath_sample_list.size(), mid + overlap));

            for (int i = left_bound; i < mid; i++) {
                auto sample     = subpath_sample_list[i];
                sample.filtered = true;
                children_samples[1].push_back(sample);
            }
            for (int i = mid; i < right_bound; i++) {
                auto sample     = subpath_sample_list[i];
                sample.filtered = true;
                children_samples[0].push_back(sample);
            }
        }

        for (int i = 0; i < 2; ++i) {
            uint32_t idx     = (uint32_t) cur_node_num - 2 + i;
            cur.children[i]  = idx;
            nodes[idx].axis  = (cur.axis + 1) % 6; // Switching axis
            nodes[idx].distr = chainDistributionFactory<Float, Spectrum>();
            if (global_sms_config.directional_struct == 1) {
                if (nodes[nodeIdx].distr != nullptr) {
                    nodes[idx].distr = std::make_shared<ChainDistribution>(*nodes[nodeIdx].distr);
                }
            }

            perf_after_filter_all += children_samples[i].size();

            if (global_sms_config.spatial_filter > 0.f) {
                std::vector<SubpathSample> tmp_samples;
                auto tmp_offset = offset;
                auto tmp_size   = size;
                tmp_offset[cur.axis] += i * 0.5 * size[cur.axis];
                tmp_size[cur.axis] *= 0.5;
                auto tmp_left_bound  = tmp_offset - tmp_size * global_sms_config.spatial_filter * 2;
                auto tmp_right_bound = tmp_offset + tmp_size + tmp_size * global_sms_config.spatial_filter * 2;
                for (auto &sample : children_samples[i]) {
                    auto p = point3f_to_6f(sample.xD, sample.xL);
                    if (all(tmp_left_bound <= p && tmp_right_bound >= p)) {
                        tmp_samples.push_back(sample);
                    }
                }
                swap(children_samples[i], tmp_samples);
            }
            perf_after_filter_safe += children_samples[i].size();
            nodes[idx].distr->load_samples(children_samples[i]);
        }
        cur.isLeaf = false;
        cur.distr  = nullptr;
    }

    Point6f point3f_to_6f(Point3f p) { return { p[0], p[1], p[2], p[0], p[1], p[2] }; }

    Vector6f vector3f_to_6f(Vector3f p) { return { p[0], p[1], p[2], p[0], p[1], p[2] }; }

    Point6f point3f_to_6f(Point3f p, Point3f q) { return { p[0], p[1], p[2], q[0], q[1], q[2] }; }

    Vector6f vector3f_to_6f(Vector3f p, Vector3f q) { return { p[0], p[1], p[2], q[0], q[1], q[2] }; }

    std::shared_ptr<ChainDistribution> chainDistribution(Point6f p) {
        Vector6f size = vector3f_to_6f(m_aabb.extents());
        p             = Point6f(p - point3f_to_6f(m_aabb.min));
        for (int i = 0; i < 6; i++) {
            p[i] /= size[i];
        }
        return m_nodes[0].chainDistribution(p, size, m_nodes);
    }

    Vector6f getVoxelSize(Point6f p) {
        Vector6f size = vector3f_to_6f(m_aabb.extents());
        p             = Point6f(p - point3f_to_6f(m_aabb.min));
        for (int i = 0; i < 6; i++) {
            p[i] /= size[i];
        }
        m_nodes[0].chainDistribution(p, size, m_nodes);
        return size;
    }

    void forEachChainDistributionParallel(std::function<void(std::shared_ptr<ChainDistribution>)> func) {
        int n_nodes = static_cast<int>(m_nodes.size());

#pragma omp parallel for
        for (int i = 0; i < n_nodes; ++i) {
            if (m_nodes[i].isLeaf) {
                func(m_nodes[i].distr);
            }
        }
    }

    void record_std(const Point6f &p, SubpathSample rec) { chainDistribution(p)->record(rec, 1.0f); }

    void record_filtered(const Point6f &p, SubpathSample rec) {
        Float volume        = 1;
        auto dTreeVoxelSize = getVoxelSize(p);
        for (int i = 0; i < 6; ++i) {
            volume *= dTreeVoxelSize[i];
        }
        // Jitter
        std::shared_ptr<ChainDistribution> jittered_chain_distr = nullptr;
        Vector6f offset                                         = dTreeVoxelSize;
        for (int i = 0; i < 6; i++)
            offset[i] *= rand() * 1.0 / RAND_MAX - 0.5f;

        Point6f origin       = point3f_to_6f(rec.xD, rec.xL) + offset;
        jittered_chain_distr = chainDistribution(origin);
        if (jittered_chain_distr) {
            jittered_chain_distr->record(rec, 1.0f);
        }
    }

    void record(const Point6f &p, SubpathSample rec, bool forceNoFilter = false) {
        if (global_sms_config.spatial_filter < 0 && !forceNoFilter) {
            record_filtered(p, rec);
        } else {
            record_std(p, rec);
        }
    }

    bool shallSplit(const MySTreeNode &node, size_t samplesRequired) {
        return m_nodes.size() < std::numeric_limits<uint32_t>::max() - 1 &&
               node.chainDistribution()->statisticalWeight() > samplesRequired;
    }

    struct StackNode {
        size_t index;
        int depth;
        Vector6f size;
        Point6f offset;
    };

    // Generate the structure of STree
    void refine_dfs(StackNode sNode, size_t SpatialStructureSTreeThreshold) {
        int cur_size = 0;
        if (m_nodes[sNode.index].chainDistribution() != nullptr) {
            cur_size = m_nodes[sNode.index].chainDistribution()->statisticalWeight();
        }

        if (m_nodes[sNode.index].isLeaf) {
            if (shallSplit(m_nodes[sNode.index], SpatialStructureSTreeThreshold)) {
                subdivide((int) sNode.index, m_nodes, sNode.size, sNode.offset);
            }
        }
        if (!m_nodes[sNode.index].isLeaf) {
            const MySTreeNode &node = m_nodes[sNode.index];
            if (cur_size >= 1000) {
#pragma omp parallel for
                for (int i = 0; i < 2; ++i) {
                    auto new_size   = sNode.size;
                    auto new_offset = sNode.offset;
                    new_size[node.axis] *= node.axis_alpha;
                    new_offset[node.axis] += new_size[node.axis] * i;
                    refine_dfs({ node.children[i], sNode.depth + 1, new_size, new_offset },
                               SpatialStructureSTreeThreshold);
                }
            } else {
                for (int i = 0; i < 2; ++i) {
                    auto new_size   = sNode.size;
                    auto new_offset = sNode.offset;
                    new_size[node.axis] *= node.axis_alpha;
                    new_offset[node.axis] += new_size[node.axis] * i;
                    refine_dfs({ node.children[i], sNode.depth + 1, new_size, new_offset },
                               SpatialStructureSTreeThreshold);
                }
            }
        }
    }

    void refine(size_t SpatialStructureSTreeThreshold) {
        m_nodes.resize(stree_nodes_limit);

        auto size   = vector3f_to_6f(m_aabb.extents());
        auto offset = point3f_to_6f(m_aabb.min);
        std::stack<StackNode> nodeIndices;
        refine_dfs({ 0, 1, size, offset }, SpatialStructureSTreeThreshold);
    }

    const AABB &aabb() const { return m_aabb; }

    int stree_thres = 1000;

    virtual void build(const std::vector<SubpathSample> &subpath_sample_list, const float spatial_radius) {
        double min_samples = 32;
        NanosecondTimer timer;

        if (global_sms_config.spatial_struct != 2) { // 2 means the original sdtree
            forEachChainDistributionParallel([&](std::shared_ptr<ChainDistribution> distr) { distr->clear(); });
        }

        // Multiplexed parameter `spatial_radius`
        if (spatial_radius > 1e4) {
            stree_thres = std::max(min_samples, spatial_radius * 1e-6 * std::pow(subpath_sample_list.size(), 0.5f));
        } else if (spatial_radius > 0) {
            stree_thres = spatial_radius;
        } else if (spatial_radius >= -1) {
            stree_thres = std::max(min_samples, std::pow(subpath_sample_list.size(), 0.5f));
        } else {
            stree_thres = -spatial_radius * std::pow(2.0, g_iter * 0.5);
        }

        stree_nodes_limit = subpath_sample_list.size() / stree_thres * 10 + 1e5;

        NanosecondTimer timer1;

        if (global_sms_config.spatial_struct != 2) { // 2 means the original sdtree
#pragma omp parallel for
            for (int i = 0; i < subpath_sample_list.size(); i++) {
                auto sample = subpath_sample_list[i];
                record(point3f_to_6f(sample.xD, sample.xL), sample, true);
            }
            std::cout << "splat using time " << timer1.value() << std::endl;
        }

        NanosecondTimer timer2;
        refine((size_t) (stree_thres));
        std::cout << "reset using time " << timer2.value() << std::endl;

        std::cout << "after refine STree size = " << m_node_size << std::endl;

        if (global_sms_config.spatial_filter < 0 || global_sms_config.spatial_struct == 2) {
            // We want to note that we are using discrete samples to simulate SDTree [Muller et al. 2019]
            // Therefore the record step is always behind the refine step
            // This is the esstential difference between [Muller et al. 2019] and our method
            forEachChainDistributionParallel([&](std::shared_ptr<ChainDistribution> distr) { distr->clear(); });
            NanosecondTimer timer1;
#pragma omp parallel for
            for (int i = 0; i < subpath_sample_list.size(); i++) {
                auto sample = subpath_sample_list[i];
                record(point3f_to_6f(sample.xD, sample.xL), sample);
            }
            std::cout << "resplat using time " << timer1.value() << std::endl;
        }

        NanosecondTimer timer3;
        forEachChainDistributionParallel([&](std::shared_ptr<ChainDistribution> distr) {
            if (distr->statisticalWeight() > 0) {
                distr->build_distribution();
            }
        });
        std::cout << "building using time " << timer3.value() << std::endl;

        perf_time_build += timer.value();
    }

    virtual std::shared_ptr<ChainDistribution> query(const Point3f xD, const Point3f xL) {
        NanosecondTimer timer;
        auto ans = chainDistribution(point3f_to_6f(xD, xL));
        perf_time_query += timer.value();
        perf_num_query++;
        return ans;
    }

    void print_stats() {
        std::cout << "STree stats perf_time_build " << std::fixed << std::setprecision(6) << perf_time_build * 1e-9
                  << std::endl;
        std::cout << "STree stats perf_time_query " << std::fixed << std::setprecision(6) << perf_time_query * 1e-9
                  << std::endl;
        std::cout << "STree stats perf_num_query " << perf_num_query << std::endl;
        std::cout << "STree stats perf_time_query_avg " << perf_time_query * 1e-9 / perf_num_query << std::endl;
    }

private:
    std::vector<MySTreeNode> m_nodes;
    AABB m_aabb;

    static inline std::atomic<unsigned long long> perf_time_build = 0; // nano sec
    static inline std::atomic<unsigned long long> perf_time_query = 0; // nano sec
    static inline std::atomic<unsigned long long> perf_num_query  = 0; // n

    static inline std::atomic<unsigned long long> perf_after_filter_all  = 0; // n
    static inline std::atomic<unsigned long long> perf_after_filter_safe = 0; // n
};

template <typename Float, typename Spectrum>
std::shared_ptr<SpatialStructure<Float, Spectrum>> spatialStructureFactory(BoundingBox<Point3> aabb) {
    if (global_sms_config.spatial_struct == 0) {
        return std::make_shared<SpatialStructureANN<Float, Spectrum>>(aabb);
    } else {
        return std::make_shared<SpatialStructureSTree<Float, Spectrum>>(aabb);
    }
    std::cout << "Undefined spatial struct" << std::endl;
    exit(1);
    return nullptr;
};

NAMESPACE_END(mitsuba)