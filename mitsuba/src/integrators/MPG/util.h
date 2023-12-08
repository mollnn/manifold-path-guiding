#pragma once
#include <atomic>
#include <cstring>
#include <enoki/morton.h>
#include <enoki/stl.h>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <mitsuba/core/distr_1d.h>
#include <mitsuba/core/profiler.h>
#include <mitsuba/core/progress.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/timer.h>
#include <mitsuba/core/util.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/film.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/microfacet.h>
#include <mitsuba/render/records.h>
#include <mitsuba/render/sampler.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/sensor.h>
#include <mitsuba/render/spiral.h>
#include <mutex>
#include <random>
#include <sstream>
#include <stack>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <thread>
#include <utility>
#include <vector>

NAMESPACE_BEGIN(mitsuba)

#define MAX_CHAIN_LENGTH 10

void set_bit(int &tau, int n) { tau |= 1 << n; }

int get_bit(int tau) {
    int num = tau;
    int pos = 0;

    if (num > 0xFFFF) {
        pos += 16;
        num >>= 16;
    }
    if (num > 0xFF) {
        pos += 8;
        num >>= 8;
    }
    if (num > 0xF) {
        pos += 4;
        num >>= 4;
    }
    if (num > 0x3) {
        pos += 2;
        num >>= 2;
    }
    if (num > 0x1) {
        pos += 1;
    }

    return pos;
}

void set_chaintype_bit(int &tau, int i, int value) {
    tau &= ~(1 << (i));
    if (value) {
        tau |= 1 << (i);
    }
}

int get_chaintype_bit(int tau, int i) { return (tau >> (i)) & 1; }

#define Point2 Point<Float, 2>
#define Point3 Point<Float, 3>
#define Point6f Point<Float, 6>
#define Vector2 Vector<Float, 2>
#define Vector3 Vector<Float, 3>
#define Vector6f Vector<Float, 6>

template <typename Float> static void add_to_atomic_float(std::atomic<Float> &var, Float val) {
    auto current = var.load();
    while (!var.compare_exchange_weak(current, current + val))
        ;
}

class NanosecondTimer {
public:
    using Unit = std::chrono::nanoseconds;

    NanosecondTimer() { start = std::chrono::system_clock::now(); }

    double value() const {
        auto now      = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<Unit>(now - start);
        return (double) duration.count();
    }

    double reset() {
        auto now      = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<Unit>(now - start);
        start         = now;
        return (double) duration.count();
    }

private:
    std::chrono::system_clock::time_point start;
};

template <typename Float, typename Spectrum> struct SubpathSample {
    MTS_IMPORT_TYPES();
    Point3f xD    = 0;
    Point3f xL    = 0;
    Point3f x1    = 0;
    Float energy  = 0;
    int bounce    = -1;
    int type      = -1;
    bool filtered = false;
};

NAMESPACE_END(mitsuba)