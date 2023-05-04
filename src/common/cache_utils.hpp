/*******************************************************************************
* Copyright 2023 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef COMMON_CACHE_UTILS_HPP
#define COMMON_CACHE_UTILS_HPP

#include <algorithm>
#include <future>
#include <memory>
#include <thread>
#include <unordered_map>

#include "oneapi/dnnl/dnnl_config.h"

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
#include "cpu/platform.hpp"
#else
#include <chrono>
#endif

#ifdef _WIN32
#include <windows.h>
#endif

#include "rw_mutex.hpp"

namespace dnnl {
namespace impl {
namespace utils {

// A key k and object o may share resources. This function moves the shared
// resources from a copy of object o into the key k. This is used to deduplicate
// data stored in cached objects.
template <typename K, typename O>
using key_merge_t = void (*)(const K &, const O &);

template <typename K, typename O, typename C,
        key_merge_t<K, O> key_merge = nullptr>
struct cache_t {
    using key_t = K;
    using object_t = O;
    using cache_object_t = C;
    using value_t = std::shared_future<cache_object_t>;
    using create_func_t = cache_object_t (&)(void *);

    virtual ~cache_t() = default;

    virtual status_t set_capacity(int capacity) = 0;
    virtual int get_capacity() const = 0;

    virtual int get_size() const = 0;

    // Returns the cached value or cache_object_t() on a miss
    virtual cache_object_t get(const key_t &key) = 0;

    // Returns the cached object associated with key, the object generated by
    // the create(create_context) function, or an empty object in case of
    // errors. The function create() is only called on a cache miss. The
    // returned object is added to the cache on a cache miss.
    cache_object_t get_or_create(
            const key_t &key, create_func_t create, void *create_context) {
        std::promise<cache_object_t> p_promise;
        // Try to get the shared future from the cache, if it's missing then a
        // shared future with no shared state is returned and the passed shared
        // future is added, otherwise a valid shared future is returned and no
        // insertion is performed.
        auto p_future = get_or_add(key, p_promise.get_future());

        if (p_future.valid()) {
            // The requested object is present in the cache or is being created
            // by another thread.
            return p_future.get();
        } else {
            // The requested object is NOT present in the cache therefore we
            // have to create it and notify the waiting threads once the
            // creation is done.
            cache_object_t cv = create(create_context);
            if (cv.status != status::success) {
                // Communicate an error.
                p_promise.set_value({nullptr, cv.status});
                // Remove the shared future from the cache because it's
                // invalidated. An invalidated shared future is the one that
                // stores a nullptr.
                remove_if_invalidated(key);
                return {nullptr, cv.status};
            } else {
                // Store the created object in the shared future and notify the
                // waiting threads.
                p_promise.set_value(cv);

                // The key_t may contains pointers that should reside within the
                // stored object. Therefore the pointers in the key may need
                // updated.
                update_entry(key, cv.get_value());
                return cv;
            }
        }
    }

protected:
    virtual value_t get_or_add(const key_t &key, const value_t &value) = 0;
    virtual void remove_if_invalidated(const key_t &key) = 0;
    virtual void update_entry(const key_t &key, const object_t &p) = 0;
    static utils::rw_mutex_t &rw_mutex() {
        static utils::rw_mutex_t mutex;
        return mutex;
    }
};

// The cache uses LRU replacement policy
template <typename K, typename O, typename C,
        key_merge_t<K, O> key_merge = nullptr>
struct lru_cache_t final : public cache_t<K, O, C, key_merge> {
    using lru_base_t = cache_t<K, O, C, key_merge>;
    using key_t = typename lru_base_t::key_t;
    using object_t = typename lru_base_t::object_t;
    using cache_object_t = typename lru_base_t::cache_object_t;
    using value_t = typename lru_base_t::value_t;
    lru_cache_t(int capacity) : capacity_(capacity) {}

    ~lru_cache_t() override {
        if (cache_mapper().empty()) return;

#if defined(_WIN32) \
        && (defined(DNNL_WITH_SYCL) || DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL)
        // The ntdll.dll library is located in system32, therefore setting
        // additional environment is not required.
        HMODULE handle = LoadLibraryExA(
                "ntdll.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!handle) {
            release_cache();
            return;
        }

        // RtlDllShutdownInProgress returns TRUE if the whole process terminates
        // and FALSE if DLL is being unloaded dynamically or if it’s called from
        // an executable.
        auto f = reinterpret_cast<BOOLEAN (*)(void)>(
                GetProcAddress(handle, "RtlDllShutdownInProgress"));
        if (!f) {
            auto ret = FreeLibrary(handle);
            assert(ret);
            MAYBE_UNUSED(ret);
            release_cache();
            return;
        }

        bool is_process_termination_in_progress = f();

        auto ret = FreeLibrary(handle);
        assert(ret);
        MAYBE_UNUSED(ret);

        if (is_process_termination_in_progress) {
            // The whole process is being terminated hence destroying content of
            // the cache cannot be done safely. However we can check all entries
            // and remove those that are not affected e.g. native CPU.
            for (auto it = cache_mapper().begin();
                    it != cache_mapper().end();) {
                if (!it->first.has_runtime_dependencies()) {
                    it = cache_mapper().erase(it);
                } else {
                    ++it;
                }
            }
            release_cache();
        } else {
            // Three scenarios possible:
            //    1. oneDNN is being dynamically unloaded
            //    2. Another dynamic library that contains statically linked
            //       oneDNN is dynamically unloaded
            //    3. oneDNN is statically linked in an executable which is done
            //       and now the process terminates In all these scenarios
            //       content of the cache can be safely destroyed.
        }
#else
            // Always destroy the content of the cache for non-Windows OSes, and
            // non-sycl and non-ocl runtimes because there is no a problem with
            // library unloading order in such cases.
#endif
    }

    cache_object_t get(const key_t &key) override {
        value_t e;
        {
            utils::lock_read_t lock_r(this->rw_mutex());
            if (capacity_ == 0) { return cache_object_t(); }
            e = get_future(key);
        }

        if (e.valid()) return e.get();
        return cache_object_t();
    }

    int get_capacity() const override {
        utils::lock_read_t lock_r(this->rw_mutex());
        return capacity_;
    };

    status_t set_capacity(int capacity) override {
        utils::lock_write_t lock_w(this->rw_mutex());
        capacity_ = capacity;
        // Check if number of entries exceeds the new capacity
        if (get_size_no_lock() > capacity_) {
            // Evict excess entries
            int n_excess_entries = get_size_no_lock() - capacity_;
            evict(n_excess_entries);
        }
        return status::success;
    }
    void set_capacity_without_clearing(int capacity) {
        utils::lock_write_t lock_w(this->rw_mutex());
        capacity_ = capacity;
    }

    int get_size() const override {
        utils::lock_read_t lock_r(this->rw_mutex());
        return get_size_no_lock();
    }

protected:
    int get_size_no_lock() const { return (int)cache_mapper().size(); }

    value_t get_or_add(const key_t &key, const value_t &value) override {
        {
            // 1. Section with shared access (read lock)
            utils::lock_read_t lock_r(this->rw_mutex());
            // Check if the cache is enabled.
            if (capacity_ == 0) { return value_t(); }
            // Check if the requested entry is present in the cache (likely
            // cache_hit)
            auto e = get_future(key);
            if (e.valid()) { return e; }
        }

        utils::lock_write_t lock_w(this->rw_mutex());
        // 2. Section with exclusive access (write lock).
        // In a multithreaded scenario, in the context of one thread the cache
        // may have changed by another thread between releasing the read lock
        // and acquiring the write lock (a.k.a. ABA problem), therefore
        // additional checks have to be performed for correctness. Double check
        // the capacity due to possible race condition
        if (capacity_ == 0) { return value_t(); }

        // Double check if the requested entry is present in the cache (unlikely
        // cache_hit).
        auto e = get_future(key);
        if (!e.valid()) {
            // If the entry is missing in the cache then add it (cache_miss)
            add(key, value);
        }
        return e;
    }

    void remove_if_invalidated(const key_t &key) override {
        utils::lock_write_t lock_w(this->rw_mutex());

        if (capacity_ == 0) { return; }

        auto it = cache_mapper().find(key);
        // The entry has been already evicted at this point
        if (it == cache_mapper().end()) { return; }

        const auto &value = it->second.value_;
        // If the entry is not invalidated
        if (value.get().is_empty()) { return; }

        // Remove the invalidated entry
        cache_mapper().erase(it);
    }

private:
    static size_t get_timestamp() {
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
        return cpu::platform::get_timestamp();
#else
        return std::chrono::steady_clock::now().time_since_epoch().count();
#endif
    }

    void update_entry(const key_t &key, const object_t &p) override {
        // Cast to void as compilers may warn about comparing compile time
        // constant function pointers with nullptr, as that is often not an
        // intended behavior
        if ((void *)key_merge == nullptr) return;

        utils::lock_write_t lock_w(this->rw_mutex());

        if (capacity_ == 0) { return; }

        // There is nothing to do in two cases:
        // 1. The requested entry is not in the cache because it has been evicted
        //    by another thread
        // 2. After the requested entry had been evicted it was inserted again
        //    by another thread
        auto it = cache_mapper().find(key);
        if (it == cache_mapper().end()
                || it->first.thread_id() != key.thread_id()) {
            return;
        }

        key_merge(it->first, p);
    }

    void evict(int n) {
        using v_t =
                typename std::unordered_map<key_t, timed_entry_t>::value_type;

        if (n == capacity_) {
            cache_mapper().clear();
            return;
        }

        for (int e = 0; e < n; e++) {
            // Find the smallest timestamp
            // TODO: revisit the eviction algorithm due to O(n) complexity, E.g.
            // maybe evict multiple entries at once.
            auto it = std::min_element(cache_mapper().begin(),
                    cache_mapper().end(),
                    [&](const v_t &left, const v_t &right) {
                        // By default, load() and operator T use sequentially
                        // consistent memory ordering, which enforces writing
                        // the timestamps into registers in the same exact order
                        // they are read from the CPU cache line. Since eviction
                        // is performed under a write lock, this order is not
                        // important, therefore we can safely use the weakest
                        // memory ordering (relaxed). This brings about a few
                        // microseconds performance improvement for default
                        // cache capacity.
                        return left.second.timestamp_.load(
                                       std::memory_order_relaxed)
                                < right.second.timestamp_.load(
                                        std::memory_order_relaxed);
                    });
            auto res = cache_mapper().erase(it->first);
            MAYBE_UNUSED(res);
            assert(res);
        }
    }
    void add(const key_t &key, const value_t &value) {
        // std::list::size() method has linear complexity. Check the cache size
        // using std::unordered_map::size();
        if (get_size_no_lock() == capacity_) {
            // Evict the least recently used entry
            evict(1);
        }

        size_t timestamp = get_timestamp();

        auto res = cache_mapper().emplace(std::piecewise_construct,
                std::forward_as_tuple(key),
                std::forward_as_tuple(value, timestamp));
        MAYBE_UNUSED(res);
        assert(res.second);
    }
    value_t get_future(const key_t &key) {
        auto it = cache_mapper().find(key);
        if (it == cache_mapper().end()) return value_t();

        size_t timestamp = get_timestamp();
        it->second.timestamp_.store(timestamp);
        // Return the entry
        return it->second.value_;
    }

    int capacity_;
    struct timed_entry_t {
        value_t value_;
        std::atomic<size_t> timestamp_;
        timed_entry_t(const value_t &value, size_t timestamp)
            : value_(value), timestamp_(timestamp) {}
    };

    std::unordered_map<key_t, timed_entry_t> &cache_mapper() {
        return cache_mapper_;
    }

    const std::unordered_map<key_t, timed_entry_t> &cache_mapper() const {
        return cache_mapper_;
    }

    // Leaks cached resources. Used to avoid issues with calling destructors
    // allocated by an already unloaded dynamic library.
    void release_cache() {
        auto t = utils::make_unique<std::unordered_map<key_t, timed_entry_t>>();
        std::swap(*t, cache_mapper_);
        t.release();
    }
    // Each entry in the cache has a corresponding key and timestamp. NOTE:
    // pairs that contain atomics cannot be stored in an unordered_map *as an
    // element*, since it invokes the copy constructor of std::atomic, which is
    // deleted.
    std::unordered_map<key_t, timed_entry_t> cache_mapper_;
};

} // namespace utils
} // namespace impl
} // namespace dnnl
#endif
