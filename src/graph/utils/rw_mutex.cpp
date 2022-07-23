/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif

#ifdef DNNL_GRAPH_SUPPORT_CXX17
#include <shared_mutex>
#endif

#include "graph/utils/compatible.hpp"
#include "graph/utils/rw_mutex.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace utils {

#ifdef DNNL_GRAPH_SUPPORT_CXX17

struct rw_mutex_t::rw_mutex_impl_t {
    using rwlock_t = std::shared_mutex;
    rwlock_t &impl() { return impl_; }

private:
    rwlock_t impl_;
};

rw_mutex_t::rw_mutex_t() {
    rw_mutex_impl_ = utils::make_unique<rw_mutex_impl_t>();
}

void rw_mutex_t::lock_read() {
    auto &impl = rw_mutex_impl_->impl();
    impl.lock_shared();
}

void rw_mutex_t::lock_write() {
    auto &impl = rw_mutex_impl_->impl();
    impl.lock();
}

void rw_mutex_t::unlock_read() {
    auto &impl = rw_mutex_impl_->impl();
    impl.unlock_shared();
}

void rw_mutex_t::unlock_write() {
    auto &impl = rw_mutex_impl_->impl();
    impl.unlock();
}

rw_mutex_t::~rw_mutex_t() {}

#else

struct rw_mutex_t::rw_mutex_impl_t {
#ifdef _WIN32
    using rwlock_t = SRWLOCK;
#else
    using rwlock_t = pthread_rwlock_t;
#endif
    rwlock_t &impl() { return impl_; }

private:
    rwlock_t impl_;
};

rw_mutex_t::rw_mutex_t() {
    rw_mutex_impl_ = utils::make_unique<rw_mutex_impl_t>();
    auto &impl = rw_mutex_impl_->impl();
#ifdef _WIN32
    InitializeSRWLock(&impl);
#else
    pthread_rwlock_init(&impl, nullptr);
#endif
}

void rw_mutex_t::lock_read() {
    auto &impl = rw_mutex_impl_->impl();
#ifdef _WIN32
    AcquireSRWLockShared(&impl);
#else
    pthread_rwlock_rdlock(&impl);
#endif
}

void rw_mutex_t::lock_write() {
    auto &impl = rw_mutex_impl_->impl();
#ifdef _WIN32
    AcquireSRWLockExclusive(&impl);
#else
    pthread_rwlock_wrlock(&impl);
#endif
}

void rw_mutex_t::unlock_read() {
    auto &impl = rw_mutex_impl_->impl();
#ifdef _WIN32
    ReleaseSRWLockShared(&impl);
#else
    pthread_rwlock_unlock(&impl);
#endif
}

void rw_mutex_t::unlock_write() {
    auto &impl = rw_mutex_impl_->impl();
#ifdef _WIN32
    ReleaseSRWLockExclusive(&impl);
#else
    pthread_rwlock_unlock(&impl);
#endif
}

rw_mutex_t::~rw_mutex_t() {
// SRW locks do not need to be explicitly destroyed
#ifndef _WIN32
    auto &impl = rw_mutex_impl_->impl();
    pthread_rwlock_destroy(&impl);
#endif
}

#endif // DNNL_GRAPH_SUPPORT_CXX17

lock_read_t::lock_read_t(rw_mutex_t &rw_mutex) : rw_mutex_(rw_mutex) {
    rw_mutex_.lock_read();
}

lock_write_t::lock_write_t(rw_mutex_t &rw_mutex) : rw_mutex_(rw_mutex) {
    rw_mutex_.lock_write();
}

lock_read_t::~lock_read_t() {
    rw_mutex_.unlock_read();
}

lock_write_t::~lock_write_t() {
    rw_mutex_.unlock_write();
}

} // namespace utils
} // namespace graph
} // namespace impl
} // namespace dnnl
