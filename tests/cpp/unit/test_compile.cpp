/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <string>

#include <gtest/gtest.h>

#include "interface/backend.hpp"

#include "backend/dnnl/common.hpp"
#include "backend/dnnl/dnnl_backend.hpp"
#include "backend/dnnl/operators/batchnorm.hpp"
#include "backend/dnnl/operators/conv.hpp"
#include "backend/dnnl/operators/eltwise.hpp"
#include "backend/dnnl/operators/inner_product.hpp"
#include "backend/dnnl/operators/layernorm.hpp"
#include "backend/dnnl/operators/matmul.hpp"
#include "backend/dnnl/operators/pool.hpp"
#include "backend/dnnl/operators/reorder.hpp"
#include "backend/dnnl/operators/softmax.hpp"

#include "unit_test_common.hpp"

#include "dnnl.h"
#include "utils.hpp"

namespace impl = dnnl::graph::impl;
namespace dnnl_impl = impl::dnnl_impl;
namespace utils = dnnl::graph::tests::unit::utils;

static dnnl_impl::kernel_registry &get_dnnl_kernel_registry() {
    return dnnl_impl::dnnl_backend::get_singleton().get_kernel_registry();
}

TEST(operator_compile, convolution_compile_fp32) {
    using dims = impl::dnnl_impl::dims;

    impl::engine_t &engine = get_engine();

    impl::op_t conv_op(impl::op_kind::Convolution);
    conv_op.set_attr<dims>("strides", dims {1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_op.set_attr<int64_t>("groups", 0);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src = utils::logical_tensor_init(
            0, {8, 3, 224, 224}, impl::data_type::f32);
    impl::logical_tensor_t weight = utils::logical_tensor_init(
            1, {16, 3, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            2, {8, 16, 222, 222}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src, weight};
    std::vector<impl::logical_tensor_t> outputs {dst};

    size_t operator_num = get_dnnl_kernel_registry().get_register_kernels_num();
    ASSERT_NE(operator_num, 0);

    auto kernel = get_dnnl_kernel_registry().create_kernel(conv_op);
    ASSERT_TRUE(kernel);

    kernel->compile(&conv_op, &engine, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);
}

TEST(operator_compile, convolution_backward_data_compile_fp32) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;

    impl::engine_t &eng = get_engine();

    impl::op_t conv_op(impl::op_kind::ConvolutionBackpropData);
    conv_op.set_attr<dims>("strides", dims {1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_op.set_attr<int64_t>("groups", 0);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t diff_src = utils::logical_tensor_init(
            0, {8, 3, 224, 224}, impl::data_type::f32);
    impl::logical_tensor_t weights = utils::logical_tensor_init(
            1, {16, 3, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t diff_dst = utils::logical_tensor_init(
            2, {8, 16, 222, 222}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {diff_dst, weights};
    std::vector<impl::logical_tensor_t> outputs {diff_src};

    size_t operator_num = get_dnnl_kernel_registry().get_register_kernels_num();
    ASSERT_NE(operator_num, 0);

    auto conv_bwd_kernel = get_dnnl_kernel_registry().create_kernel(conv_op);
    ASSERT_TRUE(conv_bwd_kernel);

    conv_bwd_kernel->compile(&conv_op, &eng, inputs, outputs);

    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::strided);
}

TEST(operator_compile, convolution_backward_weights_compile_fp32) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;

    impl::engine_t &eng = get_engine();

    impl::op_t conv_op(impl::op_kind::conv_bwd_f_biasadd_bwd);
    conv_op.set_attr<dims>("strides", dims {1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_op.set_attr<int64_t>("groups", 0);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src = utils::logical_tensor_init(
            0, {8, 3, 224, 224}, impl::data_type::f32);
    impl::logical_tensor_t diff_weights = utils::logical_tensor_init(
            2, {16, 3, 3, 3}, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t diff_bias = utils::logical_tensor_init(
            3, dims {16}, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t diff_dst = utils::logical_tensor_init(
            4, {8, 16, 222, 222}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src, diff_dst};
    std::vector<impl::logical_tensor_t> outputs {diff_weights, diff_bias};

    size_t operator_num = get_dnnl_kernel_registry().get_register_kernels_num();
    ASSERT_NE(operator_num, 0);

    auto conv_bwd_kernel = get_dnnl_kernel_registry().create_kernel(conv_op);
    ASSERT_TRUE(conv_bwd_kernel);

    conv_bwd_kernel->compile(&conv_op, &eng, inputs, outputs);

    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);
    ASSERT_EQ(outputs[1].layout_type, impl::layout_type::opaque);
}

void ref_batchnorm_fwd(impl::dim_t mb, impl::dim_t ic, impl::dim_t ih,
        impl::dim_t iw, impl::tensor_t *src, impl::tensor_t *dst,
        impl::tensor_t *scale, impl::tensor_t *shift,
        impl::tensor_t *mean = nullptr, impl::tensor_t *variance = nullptr,
        float epsilon = 0.001f,
        std::function<float(const float)> activation = nullptr,
        bool channel_last = false) {
    if (!activation) activation = [](const float v) { return v; };

    float *mean_ptr = nullptr;
    float *variance_ptr = nullptr;

    size_t ic_ = static_cast<size_t>(ic);
    size_t ih_ = static_cast<size_t>(ih);
    size_t iw_ = static_cast<size_t>(iw);

    std::unique_ptr<float[]> mean_data;
    std::unique_ptr<float[]> variance_data;
    if (!mean) {
        mean_data.reset(new float[static_cast<size_t>(ic)]);
        mean_ptr = mean_data.get();
    } else {
        mean_ptr = mean->get_data_handle<float>();
    }
    if (!variance) {
        variance_data.reset(new float[static_cast<size_t>(ic)]);
        variance_ptr = variance_data.get();
    } else {
        variance_ptr = variance->get_data_handle<float>();
    }

    float *src_ptr = src->get_data_handle<float>();

    // derives ref mean
    for (size_t channel = 0; channel < ic; ++channel) {
        mean_ptr[channel] = 0.f;
        for (size_t batch = 0; batch < mb; ++batch) {
            for (size_t h = 0; h < ih; ++h) {
                for (size_t w = 0; w < iw; ++w) {
                    size_t idx = channel_last
                            ? channel + w * ic_ + h * iw_ * ic_
                                    + batch * ih_ * iw_ * ic_
                            : w + h * iw_ + channel * ih_ * iw_
                                    + batch * ic_ * ih_ * iw_;
                    mean_ptr[channel] += src_ptr[idx];
                }
            }
        }
        mean_ptr[channel] /= static_cast<float>(mb * ih * iw);
    }
    // derives ref variance
    for (size_t channel = 0; channel < ic; ++channel) {
        variance_ptr[channel] = 0.f;
        for (size_t batch = 0; batch < mb; ++batch) {
            for (size_t h = 0; h < ih; ++h) {
                for (size_t w = 0; w < iw; ++w) {
                    size_t idx = channel_last
                            ? channel + w * ic_ + h * iw_ * ic_
                                    + batch * ih_ * iw_ * ic_
                            : w + h * iw_ + channel * ih_ * iw_
                                    + batch * ic_ * ih_ * iw_;
                    float variance_tmp = src_ptr[idx] - mean_ptr[channel];
                    variance_ptr[channel] += variance_tmp * variance_tmp;
                }
            }
        }
        variance_ptr[channel] /= static_cast<float>(mb * ih * iw);
    }

    float *dst_ptr = dst->get_data_handle<float>();
    float *scale_ptr = scale->get_data_handle<float>();
    float *shift_ptr = shift->get_data_handle<float>();
    for (size_t batch = 0; batch < mb; ++batch) {
        for (size_t channel = 0; channel < ic; ++channel) {
            for (size_t h = 0; h < ih; ++h) {
                for (size_t w = 0; w < iw; ++w) {
                    size_t idx = channel_last
                            ? channel + w * ic_ + h * iw_ * ic_
                                    + batch * ih_ * iw_ * ic_
                            : w + h * iw_ + channel * ih_ * iw_
                                    + batch * ic_ * ih_ * iw_;
                    dst_ptr[idx] = activation(scale_ptr[channel]
                                    * (src_ptr[idx] - mean_ptr[channel])
                                    / sqrtf(variance_ptr[channel] + epsilon)
                            + shift_ptr[channel]);
                }
            }
        }
    }
}

struct dnnl_graph_test_batchnorm_params {
    impl::op_kind_t op_kind;
    impl::dim_t N;
    impl::dim_t IC;
    impl::dim_t IH;
    impl::dim_t IW;
    float epsilon;
    std::string data_format;
    std::string backend_name;
    impl::data_type_t data_type;
};

class test_batchnorm_compile
    : public ::testing::TestWithParam<dnnl_graph_test_batchnorm_params> {
public:
    void TestBatchnorm(bool dims_in_order = true) {
        using dims = dnnl::graph::impl::dnnl_impl::dims;

        auto params = ::testing::TestWithParam<
                dnnl_graph_test_batchnorm_params>::GetParam();

        impl::op_t batchnorm_op(params.op_kind);
        batchnorm_op.set_attr("epsilon", params.epsilon);
        batchnorm_op.set_attr("data_format", params.data_format);

        impl::engine_t &engine = get_engine();

        // Tensor dimensions.
        const impl::dim_t N = params.N, // batch size
                IC = params.IC, // channels
                IH = params.IH, // tensor height
                IW = params.IW; // tensor width
        // Source (src) and destination (dst) tensors dimensions.
        dims src_dims;

        // |                     | == NCX?   | == NXC?   |
        // |---------------------|-----------|-----------|
        // |in-order (true)      | NCHW      | NHWC      |
        // |out-of-order (false) | NHWC      | HCHW      |
        if ((params.data_format == "NCX") ^ dims_in_order)
            src_dims = {N, IH, IW, IC};
        else
            src_dims = {N, IC, IH, IW};
        // Scale/shift tensor dimensions.
        dims scale_dims = {IC};
        dims shift_dims = {IC};

        // prepare logical tensor
        impl::logical_tensor_t src
                = utils::logical_tensor_init(0, src_dims, impl::data_type::f32);
        impl::logical_tensor_t scale = utils::logical_tensor_init(
                1, scale_dims, impl::data_type::f32);
        impl::logical_tensor_t shift = utils::logical_tensor_init(
                2, shift_dims, impl::data_type::f32);
        impl::logical_tensor_t dst
                = utils::logical_tensor_init(3, src_dims, impl::data_type::f32);

        // Allocate buffers.
        test::vector<float> src_data(static_cast<size_t>(N * IC * IH * IW));
        test::vector<float> scale_data(static_cast<size_t>(IC));
        test::vector<float> shift_data(static_cast<size_t>(IC));
        test::vector<float> dst_data(src_data.size(), 0.0);
        test::vector<float> ref_dst_data(src_data.size(), 0.0);
        // Initialize src.
        std::generate(src_data.begin(), src_data.end(), []() {
            static int i = 0;
            return std::cos(static_cast<float>(i++) / 10.f);
        });
        // Initialize scale.
        std::generate(scale_data.begin(), scale_data.end(), []() {
            static int i = 0;
            return std::sin(static_cast<float>(i++) * 2.f);
        });
        // Initialize shift.
        std::generate(shift_data.begin(), shift_data.end(), []() {
            static int i = 0;
            return std::tan(i++);
        });

        auto &op_factory = get_dnnl_kernel_registry();
        auto bn_kernel = op_factory.create_kernel(batchnorm_op);
        ASSERT_TRUE(bn_kernel);

        std::vector<impl::logical_tensor_t> inputs {src, scale, shift};
        std::vector<impl::logical_tensor_t> outputs {dst};

        // compile the bn operator
        bn_kernel->compile(&batchnorm_op, &engine, inputs, outputs);

        ASSERT_EQ(outputs[0].layout_type, impl::layout_type::strided);

        impl::tensor_t src_ts(src, src_data.data());
        impl::tensor_t scale_ts(scale, scale_data.data());
        impl::tensor_t shift_ts(shift, shift_data.data());
        impl::tensor_t dst_ts(dst, dst_data.data());
        impl::tensor_t ref_dst_ts(dst, ref_dst_data.data());

        impl::stream_t &strm = get_stream();

        if (dims_in_order) {
            bn_kernel->execute(&batchnorm_op, &strm,
                    {src_ts, scale_ts, shift_ts}, {dst_ts});
            strm.wait();

            std::function<float(const float)> activation = nullptr;
            if (params.op_kind == impl::op_kind::bn_relu) {
                activation = [](const float v) { return v < 0.f ? 0.f : v; };
            }
            ref_batchnorm_fwd(N, IC, IH, IW, &src_ts, &ref_dst_ts, &scale_ts,
                    &shift_ts, nullptr, nullptr, params.epsilon, activation,
                    params.data_format == "NXC");

            for (size_t i = 0; i < dst_data.size(); ++i) {
                ASSERT_NEAR(dst_data[i], ref_dst_data[i], 1.e-6f);
            }
        } else {
            EXPECT_THROW(bn_kernel->execute(&batchnorm_op, &strm,
                                 {src_ts, scale_ts, shift_ts}, {dst_ts}),
                    std::runtime_error);
            strm.wait();
        };
    }

    void Test() {
        TestBatchnorm(true);
        TestBatchnorm(false);
    }
};

TEST_P(test_batchnorm_compile, TestBatchnormCompile) {}

INSTANTIATE_TEST_SUITE_P(TestBatchnormCompile, test_batchnorm_compile,
        ::testing::Values(
                dnnl_graph_test_batchnorm_params {
                        impl::op_kind::BatchNormInference, 3, 3, 2, 2, 0.001f,
                        "NCX", "dnnl", impl::data_type::f32},
                dnnl_graph_test_batchnorm_params {
                        impl::op_kind::BatchNormInference, 3, 3, 2, 2, 0.001f,
                        "NXC", "dnnl", impl::data_type::f32},
                dnnl_graph_test_batchnorm_params {impl::op_kind::bn_relu, 3, 3,
                        2, 2, 0.001f, "NCX", "dnnl", impl::data_type::f32},
                dnnl_graph_test_batchnorm_params {impl::op_kind::bn_relu, 3, 3,
                        2, 2, 0.001f, "NXC", "dnnl", impl::data_type::f32}));

TEST(operator_compile, bn_compile_bwd_fp32) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;

    impl::op_t bn_op(impl::op_kind::BatchNormTrainingBackprop);
    bn_op.set_attr<float>("epsilon", 0.0);
    bn_op.set_attr<std::string>("data_format", "NCX");

    impl::engine_t &engine = get_engine();

    // Tensor dimensions.
    const impl::dim_t N = 1, // batch size
            IC = 1, // channels
            IH = 2, // tensor height
            IW = 2; // tensor width
    // Source (src) and destination (dst) tensors dimensions.
    dims src_dims = {N, IC, IH, IW};
    // Scale/shift tensor dimensions.
    dims scale_dims = {IC};
    dims shift_dims = {IC};
    dims mean_dims = {IC};
    dims varience_dims = {IC};

    // prepare logical tensor
    impl::logical_tensor_t src = utils::logical_tensor_init(
            0, src_dims, impl::data_type::f32, impl::layout_type::strided);
    impl::logical_tensor_t scale = utils::logical_tensor_init(
            1, scale_dims, impl::data_type::f32, impl::layout_type::strided);
    impl::logical_tensor_t mean = utils::logical_tensor_init(
            2, mean_dims, impl::data_type::f32, impl::layout_type::strided);
    impl::logical_tensor_t varience = utils::logical_tensor_init(
            3, varience_dims, impl::data_type::f32, impl::layout_type::strided);
    impl::logical_tensor_t diff_dst = utils::logical_tensor_init(
            4, src_dims, impl::data_type::f32, impl::layout_type::strided);
    impl::logical_tensor_t diff_src = utils::logical_tensor_init(
            5, src_dims, impl::data_type::f32, impl::layout_type::strided);
    impl::logical_tensor_t diff_scale = utils::logical_tensor_init(
            6, scale_dims, impl::data_type::f32, impl::layout_type::strided);
    impl::logical_tensor_t diff_shift = utils::logical_tensor_init(
            7, shift_dims, impl::data_type::f32, impl::layout_type::strided);

    // Allocate buffers.
    test::vector<float> src_data {1.0, 2.0, 3.0, 4.0};
    test::vector<float> scale_data {2.0, 0.0, 0.0, 0.0};
    test::vector<float> mean_data {2.5, 0.0, 0.0, 0.0};
    test::vector<float> varience_data {1.25, 0.0, 0.0, 0.0};
    test::vector<float> diff_dst_data {0.1f, 0.1f, 0.2f, 0.2f};
    test::vector<float> diff_src_data(src_data.size(), 0.0);
    test::vector<float> diff_scale_data(scale_data.size(), 0.0);
    test::vector<float> diff_shift_data(scale_data.size(), 0.0);

    // expected results
    test::vector<float> ref_diff_src_data {
            0.17888544f, 0.17888544f, 0.35777088f, 0.35777088f};
    test::vector<float> ref_diff_scale_data {0.178885f, 0.0, 0.0, 0.0};
    test::vector<float> ref_diff_shift_data {0.6f, 0.0, 0.0, 0.0};

    auto &op_factory = get_dnnl_kernel_registry();
    auto bn_kernel = op_factory.create_kernel(bn_op);
    ASSERT_TRUE(bn_kernel);

    std::vector<impl::logical_tensor_t> inputs {
            src, diff_dst, scale, mean, varience};
    std::vector<impl::logical_tensor_t> outputs {
            diff_src, diff_scale, diff_shift};

    // compile the bn operator
    bn_kernel->compile(&bn_op, &engine, inputs, outputs);

    impl::tensor_t src_ts(src, src_data.data());
    impl::tensor_t scale_ts(scale, scale_data.data());
    impl::tensor_t mean_ts(mean, mean_data.data());
    impl::tensor_t varience_ts(varience, varience_data.data());
    impl::tensor_t diff_dst_ts(diff_dst, diff_dst_data.data());
    impl::tensor_t diff_src_ts(diff_src, diff_src_data.data());
    impl::tensor_t diff_scale_ts(diff_scale, diff_scale_data.data());
    impl::tensor_t diff_shift_ts(diff_shift, diff_shift_data.data());

    impl::stream_t &strm = get_stream();
    bn_kernel->execute(&bn_op, &strm,
            {src_ts, diff_dst_ts, scale_ts, mean_ts, varience_ts},
            {diff_src_ts, diff_scale_ts, diff_shift_ts});
    strm.wait();
    for (size_t i = 0; i < diff_src_data.size(); ++i) {
        auto absolute_error
                = std::fabs(diff_src_data[i] - ref_diff_src_data[i]);
        ASSERT_TRUE(absolute_error < 0.001);
    }
    for (size_t i = 0; i < diff_scale_data.size(); ++i) {
        auto absolute_error
                = std::fabs(diff_scale_data[i] - ref_diff_scale_data[i]);
        ASSERT_TRUE(absolute_error < 0.001);
    }
    for (size_t i = 0; i < diff_shift_data.size(); ++i) {
        auto absolute_error
                = std::fabs(diff_shift_data[i] - ref_diff_shift_data[i]);
        ASSERT_TRUE(absolute_error < 0.001);
    }
}

TEST(operator_kernel, relu) {
    test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0};
    test::vector<float> ref_dst {0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.5, 2.0};
    test::vector<float> dst(src.size(), 0.0);

    impl::op_t relu_op(impl::op_kind::ReLU);

    auto &op_factory = get_dnnl_kernel_registry();
    auto relu_kernel = op_factory.create_kernel(relu_op);
    ASSERT_TRUE(relu_kernel);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);

    impl::engine_t &eng = get_engine();
    // compile the relu operator
    relu_kernel->compile(&relu_op, &eng, {src_lt}, {dst_lt});

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    impl::stream_t &strm = get_stream();
    relu_kernel->execute(&relu_op, &strm, {src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, relu_backward) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0};
    test::vector<float> diff_dst {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    test::vector<float> ref_diff_src {
            0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 6.0, 7.0, 8.0};
    test::vector<float> diff_src(src.size(), 0.0);

    impl::op_t relu_op(impl::op_kind::ReLUBackprop);

    auto &op_factory = get_dnnl_kernel_registry();
    auto relu_kernel = op_factory.create_kernel(relu_op);
    ASSERT_TRUE(relu_kernel);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t diff_src_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t diff_dst_lt
            = utils::logical_tensor_init(2, {1, 3, 3}, impl::data_type::f32);

    // compile the relu backward operator
    relu_kernel->compile(&relu_op, &eng, {diff_dst_lt, src_lt}, {diff_src_lt});

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t diff_src_ts(diff_src_lt, diff_src.data());
    impl::tensor_t diff_dst_ts(diff_dst_lt, diff_dst.data());

    impl::stream_t &strm = get_stream();
    relu_kernel->execute(&relu_op, &strm, {diff_dst_ts, src_ts}, {diff_src_ts});
    strm.wait();
    for (size_t i = 0; i < diff_src.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_src[i], ref_diff_src[i]);
    }
}

TEST(operator_kernel, gelu) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0};
    test::vector<float> ref_dst {-0.0455001f, -0.10021085f, -0.15865527f,
            -0.15426877f, 0.f, 0.3457312f, 0.84134465f, 1.399789f, 1.9544998f};
    test::vector<float> dst(src.size(), 0.0);

    impl::op_t gelu_op(impl::op_kind::GELU);

    auto &op_factory = get_dnnl_kernel_registry();
    auto gelu_kernel = op_factory.create_kernel(gelu_op);
    ASSERT_TRUE(gelu_kernel);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);

    gelu_kernel->compile(&gelu_op, &eng, {src_lt}, {dst_lt});

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    impl::stream_t &strm = get_stream();
    gelu_kernel->execute(&gelu_op, &strm, {src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_NEAR(dst[i], ref_dst[i], 1.e-6f);
    }
}

TEST(operator_kernel, gelu_backward) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0};
    test::vector<float> diff_dst {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    test::vector<float> ref_diff_src {0.0f, -0.1274692f, -0.1666309f,
            0.3975146f, 2.0f, 4.3374756f, 6.4998928f, 7.8922843f, 8.6818544f};
    test::vector<float> diff_src(src.size(), 0.0);

    impl::op_t gelu_op(impl::op_kind::GELUBackprop);

    auto &op_factory = get_dnnl_kernel_registry();
    auto gelu_kernel = op_factory.create_kernel(gelu_op);
    ASSERT_TRUE(gelu_kernel);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t diff_src_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t diff_dst_lt
            = utils::logical_tensor_init(2, {1, 3, 3}, impl::data_type::f32);

    // compile the helu backward operator
    gelu_kernel->compile(&gelu_op, &eng, {diff_dst_lt, src_lt}, {diff_src_lt});

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t diff_src_ts(diff_src_lt, diff_src.data());
    impl::tensor_t diff_dst_ts(diff_dst_lt, diff_dst.data());

    impl::stream_t &strm = get_stream();
    gelu_kernel->execute(&gelu_op, &strm, {diff_dst_ts, src_ts}, {diff_src_ts});
    strm.wait();
    for (size_t i = 0; i < diff_src.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_src[i], ref_diff_src[i]);
    }
}

TEST(operator_kernel, add) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> src1 {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> ref_dst {3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t add_op(impl::op_kind::Add);

    auto &op_factory = get_dnnl_kernel_registry();
    auto add_kernel = op_factory.create_kernel(add_op);
    ASSERT_TRUE(add_kernel);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {1, 3, 3}, impl::data_type::f32, impl::layout_type::any);
    // compile the add operator
    std::vector<impl::logical_tensor_t> inputs {src0_lt, src1_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};
    add_kernel->compile(&add_op, &eng, inputs, outputs);

    ASSERT_EQ(dst_lt.layout_type, impl::layout_type::any);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src0_ts(src0_lt, src0.data());
    impl::tensor_t src1_ts(src1_lt, src1.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());

    impl::stream_t &strm = get_stream();
    add_kernel->execute(&add_op, &strm, {src0_ts, src1_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }

    // the second iteration
    test::vector<float> src0_2nd {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> src1_2nd {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> ref_dst_2nd {
            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> dst_2nd(src0_2nd.size(), 0.0);

    impl::tensor_t src0_2nd_ts(src0_lt, src0_2nd.data());
    impl::tensor_t src1_2nd_ts(src1_lt, src1_2nd.data());
    impl::tensor_t dst_2nd_ts(outputs[0], dst_2nd.data());
    add_kernel->execute(
            &add_op, &strm, {src0_2nd_ts, src1_2nd_ts}, {dst_2nd_ts});
    strm.wait();

    for (size_t i = 0; i < src0_2nd.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_2nd[i], ref_dst_2nd[i]);
    }
}

TEST(operator_kernel, different_format_add) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> src1 {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    test::vector<float> ref_dst {2.0, 5.0, 8.0, 3.0, 6.0, 9.0, 4.0, 7.0, 10.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t add_op(impl::op_kind::Add);

    auto &op_factory = get_dnnl_kernel_registry();
    auto add_kernel = op_factory.create_kernel(add_op);
    ASSERT_TRUE(add_kernel);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt = utils::logical_tensor_init(
            1, {1, 1, 3, 3}, {3, 3, 1, 3}, impl::data_type::f32); // abdc format
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 1, 3, 3}, impl::data_type::f32);
    // compile the add operator
    add_kernel->compile(&add_op, &eng, {src0_lt, src1_lt}, {dst_lt});

    impl::tensor_t src0_ts(src0_lt, src0.data());
    impl::tensor_t src1_ts(src1_lt, src1.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    impl::stream_t &strm = get_stream();
    add_kernel->execute(&add_op, &strm, {src0_ts, src1_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, broadcast_add) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> src1 {
            2.0, 2.0, 2.0}; // bianary op's src1 support broadcast
    test::vector<float> ref_dst {3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t broadcast_add_op(impl::op_kind::Add);

    auto &op_factory = get_dnnl_kernel_registry();
    auto broadcast_add_kernel = op_factory.create_kernel(broadcast_add_op);
    ASSERT_TRUE(broadcast_add_kernel);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 2, 3, 3}, impl::data_type::f32);
    // src1 will first be unsequeeze to {1,1,1,3} and then broadcast
    // to {1,2,3,3}
    impl::logical_tensor_t src1_lt = utils::logical_tensor_init(
            1, std::vector<impl::dim_t> {3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 2, 3, 3}, impl::data_type::f32);

    // compile the add operator
    broadcast_add_kernel->compile(
            &broadcast_add_op, &eng, {src0_lt, src1_lt}, {dst_lt});

    impl::tensor_t src0_ts(src0_lt, src0.data());
    impl::tensor_t src1_ts(src1_lt, src1.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    impl::stream_t &strm = get_stream();
    broadcast_add_kernel->execute(
            &broadcast_add_op, &strm, {src0_ts, src1_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, reversed_different_format_broadcast_add) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src1 {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> src0 {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    test::vector<float> ref_dst {2.0, 5.0, 8.0, 3.0, 6.0, 9.0, 4.0, 7.0, 10.0,
            2.0, 5.0, 8.0, 3.0, 6.0, 9.0, 4.0, 7.0, 10.0};
    test::vector<float> dst(src1.size(), 0.0);

    impl::op_t broadcast_add_op(impl::op_kind::Add);

    auto &op_factory = get_dnnl_kernel_registry();
    auto broadcast_add_kernel = op_factory.create_kernel(broadcast_add_op);
    ASSERT_TRUE(broadcast_add_kernel);

    // we reverse the order of src0 and src1
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(0, {1, 2, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src0_lt = utils::logical_tensor_init(
            1, {3, 3}, {1, 3}, impl::data_type::f32); // ba format
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 2, 3, 3}, impl::data_type::f32);

    // compile the add operator
    broadcast_add_kernel->compile(
            &broadcast_add_op, &eng, {src0_lt, src1_lt}, {dst_lt});

    impl::tensor_t src0_ts(src0_lt, src0.data());
    impl::tensor_t src1_ts(src1_lt, src1.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    impl::stream_t &strm = get_stream();
    broadcast_add_kernel->execute(
            &broadcast_add_op, &strm, {src0_ts, src1_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, bias_add) {
    impl::engine_t &eng = get_engine();
    auto &op_factory = get_dnnl_kernel_registry();

    test::vector<float> src {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> bias {1.0, 2.0};
    test::vector<float> ref_dst1 {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0};
    test::vector<float> ref_dst2 {2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0,
            3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0};
    test::vector<float> dst(src.size(), 0.0);

    std::vector<std::vector<impl::dim_t>> src_shapes {
            {1, 2, 3, 3}, {1, 3, 3, 2}};
    std::vector<std::vector<impl::dim_t>> dst_shapes {
            {1, 2, 3, 3}, {1, 3, 3, 2}};
    std::vector<impl::dim_t> bias_shape {2};
    std::vector<std::string> data_formats {"NCX", "NXC"};

    for (size_t i = 0; i < data_formats.size(); i++) {
        impl::op_t bias_add_op(impl::op_kind::BiasAdd);
        bias_add_op.set_attr<std::string>("data_format", data_formats[i]);

        auto bias_add_kernel = op_factory.create_kernel(bias_add_op);
        ASSERT_TRUE(bias_add_kernel);

        impl::logical_tensor_t src_lt = utils::logical_tensor_init(
                0, src_shapes[i], impl::data_type::f32);
        impl::logical_tensor_t bias_lt = utils::logical_tensor_init(
                1, bias_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
                2, dst_shapes[i], impl::data_type::f32);

        // compile the add operator
        bias_add_kernel->compile(
                &bias_add_op, &eng, {src_lt, bias_lt}, {dst_lt});

        impl::tensor_t src_ts(src_lt, src.data());
        impl::tensor_t bias_ts(bias_lt, bias.data());
        impl::tensor_t dst_ts(dst_lt, dst.data());

        impl::stream_t &strm = get_stream();
        bias_add_kernel->execute(
                &bias_add_op, &strm, {src_ts, bias_ts}, {dst_ts});
        strm.wait();

        if (i == 0) {
            for (size_t i = 0; i < src.size(); ++i) {
                ASSERT_FLOAT_EQ(dst[i], ref_dst1[i]);
            }
        } else {
            for (size_t i = 0; i < src.size(); ++i) {
                ASSERT_FLOAT_EQ(dst[i], ref_dst2[i]);
            }
        }
    }
}

TEST(operator_kernel, mul) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> src1 {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> ref_dst {4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t mul_op(impl::op_kind::Multiply);

    auto &op_factory = get_dnnl_kernel_registry();
    auto mul_kernel = op_factory.create_kernel(mul_op);
    ASSERT_TRUE(mul_kernel);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 3, 3}, impl::data_type::f32);

    // compile the add operator
    mul_kernel->compile(&mul_op, &eng, {src0_lt, src1_lt}, {dst_lt});

    impl::tensor_t src0_ts(src0_lt, src0.data());
    impl::tensor_t src1_ts(src1_lt, src1.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    impl::stream_t &strm = get_stream();
    mul_kernel->execute(&mul_op, &strm, {src0_ts, src1_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, min) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0};
    test::vector<float> src1 {1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0};
    test::vector<float> ref_dst {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t max_op(impl::op_kind::Minimum);

    auto &op_factory = get_dnnl_kernel_registry();
    auto max_kernel = op_factory.create_kernel(max_op);
    ASSERT_TRUE(max_kernel);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 3, 3}, impl::data_type::f32);

    // compile the add operator
    max_kernel->compile(&max_op, &eng, {src0_lt, src1_lt}, {dst_lt});

    impl::tensor_t src0_ts(src0_lt, src0.data());
    impl::tensor_t src1_ts(src1_lt, src1.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    impl::stream_t &strm = get_stream();
    max_kernel->execute(&max_op, &strm, {src0_ts, src1_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, max) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0};
    test::vector<float> src1 {1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0};
    test::vector<float> ref_dst {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t min_op(impl::op_kind::Maximum);

    auto &op_factory = get_dnnl_kernel_registry();
    auto min_kernel = op_factory.create_kernel(min_op);
    ASSERT_TRUE(min_kernel);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 3, 3}, impl::data_type::f32);

    // compile the add operator
    min_kernel->compile(&min_op, &eng, {src0_lt, src1_lt}, {dst_lt});

    impl::tensor_t src0_ts(src0_lt, src0.data());
    impl::tensor_t src1_ts(src1_lt, src1.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    impl::stream_t &strm = get_stream();
    min_kernel->execute(&min_op, &strm, {src0_ts, src1_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, matmul_compile_fwd_fp32) {
    impl::op_t matmul_op(impl::op_kind::MatMul);
    matmul_op.set_attr<bool>("transpose_b", true);
    impl::engine_t &engine = get_engine();

    test::vector<float> src_data {-2.0, -1.5};
    test::vector<float> weight_data {-2.0, -1.5};
    test::vector<float> ref_dst_data {6.25};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {2}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            2, {1, 1}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src, weight};
    std::vector<impl::logical_tensor_t> outputs {dst};

    auto &op_factory = get_dnnl_kernel_registry();
    auto matmul_kernel = op_factory.create_kernel(matmul_op);
    matmul_kernel->compile(&matmul_op, &engine, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src, src_data.data());
    impl::tensor_t weight_ts(weight, weight_data.data());
    impl::tensor_t dst_ts(outputs[0], dst_data.data());

    impl::stream_t &strm = get_stream();
    matmul_kernel->execute(&matmul_op, &strm, {src_ts, weight_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, matmul_compile_fwd_f16f16f16) {
    impl::op_t matmul_op(impl::op_kind::MatMul);

    impl::engine_t &engine = get_engine();
    SKIP_IF(engine.kind() != impl::engine_kind::gpu,
            "Skip fp16 test for non-GPU device.");

    // in real use case, the data type should be half
    test::vector<float> src_data {-2.0, -1.5, -1.0, -0.5, 0.0, 0.5};
    test::vector<float> weight_data {
            -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, -2.0, -1.5};
    test::vector<float> dst_data(12, 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {2}, impl::data_type::f16);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2}, impl::data_type::f16);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            2, {1, 1}, impl::data_type::f16, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src, weight};
    std::vector<impl::logical_tensor_t> outputs {dst};

    auto &op_factory = get_dnnl_kernel_registry();
    auto matmul_kernel = op_factory.create_kernel(matmul_op);
    matmul_kernel->compile(&matmul_op, &engine, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src, src_data.data());
    impl::tensor_t weight_ts(weight, weight_data.data());
    impl::tensor_t dst_ts(outputs[0], dst_data.data());

    impl::stream_t &strm = get_stream();
    matmul_kernel->execute(&matmul_op, &strm, {src_ts, weight_ts}, {dst_ts});
    strm.wait();
}

TEST(operator_kernel, matmul_compile_fwd_bf16bf16bf16) {
    impl::op_t matmul_op(impl::op_kind::MatMul);

    impl::engine_t &engine = get_engine();
    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF(isa < dnnl_cpu_isa_avx512_core
                    && engine.kind() == impl::engine_kind::cpu,
            "Skip bf16 examples for systems that do not support avx512_core.");

    // in real use case, the data type should be half
    test::vector<float> src_data {-2.0, -1.5, -1.0, -0.5, 0.0, 0.5};
    test::vector<float> weight_data {
            -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, -2.0, -1.5};
    test::vector<float> dst_data(12, 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {2}, impl::data_type::bf16);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2}, impl::data_type::bf16);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            2, {1, 1}, impl::data_type::bf16, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src, weight};
    std::vector<impl::logical_tensor_t> outputs {dst};

    auto &op_factory = get_dnnl_kernel_registry();
    auto matmul_kernel = op_factory.create_kernel(matmul_op);
    matmul_kernel->compile(&matmul_op, &engine, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src, src_data.data());
    impl::tensor_t weight_ts(weight, weight_data.data());
    impl::tensor_t dst_ts(outputs[0], dst_data.data());

    impl::stream_t &strm = get_stream();
    matmul_kernel->execute(&matmul_op, &strm, {src_ts, weight_ts}, {dst_ts});
    strm.wait();
}

static size_t product(std::vector<int64_t> &in) {
    if (in.empty()) return 0;
    int64_t prod = std::accumulate(in.begin(), in.end(),
            static_cast<int64_t>(1), std::multiplies<int64_t>());
    return static_cast<size_t>(prod);
}

TEST(operator_kernel, matmul_ndx2d) {
    impl::op_t matmul_op(impl::op_kind::MatMul);
    impl::engine_t &engine = get_engine();
    auto &op_factory = get_dnnl_kernel_registry();
    auto matmul_kernel = op_factory.create_kernel(matmul_op);

    std::vector<std::vector<int64_t>> weight_shapes {{16}, {16, 1}};
    std::vector<std::vector<int64_t>> src_shapes {
            {4, 2, 16}, {6, 4, 2, 16}, {8, 6, 4, 2, 16}};
    std::vector<std::vector<int64_t>> dst_shapes {
            {4, 2, 1}, {6, 4, 2, 1}, {8, 6, 4, 2, 1}};

    // Initialize
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0f, 1.0f);

    for (size_t w_idx = 0; w_idx < weight_shapes.size(); ++w_idx) {
        for (size_t idx = 0; idx < src_shapes.size(); idx++) {
            auto src_shape = src_shapes[idx];
            auto dst_shape = dst_shapes[idx];

            test::vector<float> src_data(product(src_shape));
            test::vector<float> weight_data(product(weight_shapes[w_idx]));
            test::vector<float> dst_data(product(dst_shape));
            test::vector<float> ref_dst_data(product(dst_shape), 0.0);

            std::generate(src_data.begin(), src_data.end(),
                    [&]() { return distribution(generator); });
            std::generate(weight_data.begin(), weight_data.end(),
                    [&]() { return distribution(generator); });

            // prepare logical tensor
            impl::logical_tensor_t src = utils::logical_tensor_init(
                    0, src_shape, impl::data_type::f32);
            impl::logical_tensor_t weight = utils::logical_tensor_init(
                    1, weight_shapes[w_idx], impl::data_type::f32);
            impl::logical_tensor_t dst = utils::logical_tensor_init(
                    2, dst_shape, impl::data_type::f32, impl::layout_type::any);

            std::vector<impl::logical_tensor_t> inputs {src, weight};
            std::vector<impl::logical_tensor_t> outputs {dst};

            matmul_kernel->compile(&matmul_op, &engine, inputs, outputs);
            ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

            impl::tensor_t src_ts(src, src_data.data());
            impl::tensor_t weight_ts(weight, weight_data.data());
            impl::tensor_t dst_ts(outputs[0], dst_data.data());

            impl::stream_t &strm = get_stream();
            matmul_kernel->execute(
                    &matmul_op, &strm, {src_ts, weight_ts}, {dst_ts});
            strm.wait();

            // compute the ref results
            size_t M = product(src_shape)
                    / static_cast<size_t>(src_shape.back());
            size_t K = static_cast<size_t>(src_shape.back());
            size_t N = weight_shapes[w_idx].size() == 1
                    ? static_cast<size_t>(1)
                    : static_cast<size_t>(weight_shapes[w_idx].back());
            // size_t N = static_cast<size_t>(1);
            for (size_t m = 0; m < M; m++) {
                for (size_t n = 0; n < N; n++) {
                    for (size_t k = 0; k < K; k++)
                        ref_dst_data[m * N + n]
                                += src_data[m * K + k] * weight_data[k * N + n];
                }
            }

            // FIXME(qun) the max error will be bigger than 1e-6
            for (size_t i = 0; i < ref_dst_data.size(); ++i) {
                ASSERT_NEAR(dst_data[i], ref_dst_data[i], 3e-6f);
            }
        }
    }
}

TEST(operator_kernel, matmul_3dx3d) {
    impl::op_t matmul_op(impl::op_kind::MatMul);
    impl::engine_t &engine = get_engine();

    test::vector<float> src_data {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> weight_data {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> ref_dst_data {2.0, 2.0, 2.0, 2.0};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {4, 1, 2}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {4, 2, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            2, {4, 1, 1}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src, weight};
    std::vector<impl::logical_tensor_t> outputs {dst};

    auto &op_factory = get_dnnl_kernel_registry();
    auto matmul_kernel = op_factory.create_kernel(matmul_op);
    matmul_kernel->compile(&matmul_op, &engine, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src, src_data.data());
    impl::tensor_t weight_ts(weight, weight_data.data());
    impl::tensor_t dst_ts(outputs[0], dst_data.data());

    impl::stream_t &strm = get_stream();
    matmul_kernel->execute(&matmul_op, &strm, {src_ts, weight_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, matmul_relu_fusion) {
    impl::op_t matmul_relu_op(impl::op_kind::matmul_relu);

    impl::engine_t &engine = get_engine();

    test::vector<float> src_data {-2.0, -1.5};
    test::vector<float> weight_data {2.0, 1.5};
    test::vector<float> ref_dst_data {0.0};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {1, 2}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            2, {1, 1}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src, weight};
    std::vector<impl::logical_tensor_t> outputs {dst};

    auto &op_factory = get_dnnl_kernel_registry();
    auto matmul_relu_kernel = op_factory.create_kernel(matmul_relu_op);

    matmul_relu_kernel->compile(&matmul_relu_op, &engine, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src, src_data.data());
    impl::tensor_t weight_ts(weight, weight_data.data());
    impl::tensor_t dst_ts(outputs[0], dst_data.data());

    impl::stream_t &strm = get_stream();
    matmul_relu_kernel->execute(
            &matmul_relu_op, &strm, {src_ts, weight_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, matmul_bias_fusion) {
    impl::op_t matmul_op(impl::op_kind::matmul_bias);

    impl::engine_t &engine = get_engine();

    test::vector<float> src_data {-2.0, -1.5, 3.0, 0.5};
    test::vector<float> weight_data {-2.0, -1.5, 1.0, 1.0};
    test::vector<float> bias_data {1.0};
    test::vector<float> ref_dst_data {7.25, 4.5};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 1, 2}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 2, 1}, impl::data_type::f32);
    impl::logical_tensor_t bias
            = utils::logical_tensor_init(2, {1}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            3, {2, 1, 1}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src, weight, bias};
    std::vector<impl::logical_tensor_t> outputs {dst};

    auto &op_factory = get_dnnl_kernel_registry();
    auto matmul_kernel = op_factory.create_kernel(matmul_op);
    matmul_kernel->compile(&matmul_op, &engine, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src, src_data.data());
    impl::tensor_t weight_ts(weight, weight_data.data());
    impl::tensor_t bias_ts(bias, bias_data.data());
    impl::tensor_t dst_ts(outputs[0], dst_data.data());

    impl::stream_t &strm = get_stream();
    matmul_kernel->execute(
            &matmul_op, &strm, {src_ts, weight_ts, bias_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, matmul_sum_fusion_broadcast_1d) {
    impl::op_t matmul_op(impl::op_kind::matmul_add);

    test::vector<float> src_data {-2.0, -1.5, 3.0, 0.5};
    test::vector<float> weight_data {-2.0, -1.5, 1.0, 1.0};
    test::vector<float> add_src1_data {1.0, 1.0};
    test::vector<float> ref_dst_data {5.0, 4.0, 4.0, 3.25, 4.0, 4.0, 1.5, 1.5};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 2, 1}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 1, 2}, impl::data_type::f32);
    impl::logical_tensor_t add_src1
            = utils::logical_tensor_init(2, {2}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            3, {2, 2, 2}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src, weight, add_src1};
    std::vector<impl::logical_tensor_t> outputs {dst};

    impl::engine_t &engine = get_engine();
    auto &op_factory = get_dnnl_kernel_registry();
    auto matmul_kernel = op_factory.create_kernel(matmul_op);
    matmul_kernel->compile(&matmul_op, &engine, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src, src_data.data());
    impl::tensor_t weight_ts(weight, weight_data.data());
    impl::tensor_t add_src1_ts(add_src1, add_src1_data.data());
    impl::tensor_t dst_ts(outputs[0], dst_data.data());

    impl::stream_t &strm = get_stream();
    matmul_kernel->execute(
            &matmul_op, &strm, {src_ts, weight_ts, add_src1_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, matmul_sum_fusion) {
    impl::op_t matmul_op(impl::op_kind::matmul_add);

    test::vector<float> src_data {-2.0, -1.5, 3.0, 0.5};
    test::vector<float> weight_data {-2.0, -1.5, 1.0, 1.0};
    test::vector<float> add_src1_data {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> ref_dst_data {5.0, 4.0, 4.0, 3.25, 4.0, 4.0, 1.5, 1.5};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 2, 1}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 1, 2}, impl::data_type::f32);
    impl::logical_tensor_t add_src1
            = utils::logical_tensor_init(2, {2, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            3, {2, 2, 2}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src, weight, add_src1};
    std::vector<impl::logical_tensor_t> outputs {dst};

    impl::engine_t &engine = get_engine();
    auto &op_factory = get_dnnl_kernel_registry();
    auto matmul_kernel = op_factory.create_kernel(matmul_op);
    matmul_kernel->compile(&matmul_op, &engine, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src, src_data.data());
    impl::tensor_t weight_ts(weight, weight_data.data());
    impl::tensor_t add_src1_ts(add_src1, add_src1_data.data());
    impl::tensor_t dst_ts(outputs[0], dst_data.data());

    impl::stream_t &strm = get_stream();
    matmul_kernel->execute(
            &matmul_op, &strm, {src_ts, weight_ts, add_src1_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, matmul_sum_gelu_fusion) {
    impl::op_t matmul_op(impl::op_kind::matmul_add_gelu);

    impl::engine_t &engine = get_engine();

    test::vector<float> src_data {1.0, 1.0, 1.0, 1.0};
    test::vector<float> weight_data {0.5, 0.5, 0.5, 0.5};
    test::vector<float> other_data {1.0};
    test::vector<float> ref_dst_data {1.9544998f, 1.9544998f};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 1, 2}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 2, 1}, impl::data_type::f32);
    impl::logical_tensor_t other
            = utils::logical_tensor_init(2, {1}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            3, {2, 1, 1}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src, weight, other};
    std::vector<impl::logical_tensor_t> outputs {dst};

    auto &op_factory = get_dnnl_kernel_registry();
    auto matmul_kernel = op_factory.create_kernel(matmul_op);
    matmul_kernel->compile(&matmul_op, &engine, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src, src_data.data());
    impl::tensor_t weight_ts(weight, weight_data.data());
    impl::tensor_t other_ts(other, other_data.data());
    impl::tensor_t dst_ts(outputs[0], dst_data.data());

    impl::stream_t &strm = get_stream();
    matmul_kernel->execute(
            &matmul_op, &strm, {src_ts, weight_ts, other_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, matmul_sum_relu_fusion) {
    impl::op_t matmul_op(impl::op_kind::matmul_add_relu);
    impl::engine_t &engine = get_engine();

    test::vector<float> src_data {1.0, 2.0, 3.0, 4.0};
    test::vector<float> weight_data {-1.0, -2.0, 3.0, 4.0};
    test::vector<float> other_data {2.5};
    test::vector<float> ref_dst_data {0, 27.5f};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 1, 2}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 2, 1}, impl::data_type::f32);
    impl::logical_tensor_t other
            = utils::logical_tensor_init(2, {1}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            3, {2, 1, 1}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src, weight, other};
    std::vector<impl::logical_tensor_t> outputs {dst};

    auto &op_factory = get_dnnl_kernel_registry();
    auto matmul_kernel = op_factory.create_kernel(matmul_op);
    matmul_kernel->compile(&matmul_op, &engine, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src, src_data.data());
    impl::tensor_t weight_ts(weight, weight_data.data());
    impl::tensor_t other_ts(other, other_data.data());
    impl::tensor_t dst_ts(outputs[0], dst_data.data());

    impl::stream_t &strm = get_stream();
    matmul_kernel->execute(
            &matmul_op, &strm, {src_ts, weight_ts, other_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, matmul_bias_relu_fusion) {
    impl::op_t matmul_op(impl::op_kind::matmul_bias_relu);
    impl::engine_t &engine = get_engine();

    test::vector<float> src_data {-2.0, -1.5};
    test::vector<float> weight_data {2.0, -1.5};
    test::vector<float> bias_data {1.0};
    test::vector<float> ref_dst_data {0.0};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {1, 2}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 1}, impl::data_type::f32);
    impl::logical_tensor_t bias
            = utils::logical_tensor_init(2, {1}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            3, {1, 1}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src, weight, bias};
    std::vector<impl::logical_tensor_t> outputs {dst};

    auto &op_factory = get_dnnl_kernel_registry();
    auto matmul_kernel = op_factory.create_kernel(matmul_op);
    matmul_kernel->compile(&matmul_op, &engine, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src, src_data.data());
    impl::tensor_t weight_ts(weight, weight_data.data());
    impl::tensor_t bias_ts(bias, bias_data.data());
    impl::tensor_t dst_ts(outputs[0], dst_data.data());

    impl::stream_t &strm = get_stream();
    matmul_kernel->execute(
            &matmul_op, &strm, {src_ts, weight_ts, bias_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, matmul_bias_gelu_fusion) {
    impl::op_t matmul_op(impl::op_kind::matmul_bias_gelu);
    impl::engine_t &engine = get_engine();

    test::vector<float> src_data {1.0, 1.0, 1.0, 1.0};
    test::vector<float> weight_data {0.5, 0.5, 0.5, 0.5};
    test::vector<float> bias_data {1.0};
    test::vector<float> ref_dst_data {1.9544998f, 1.9544998f};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 1, 2}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 2, 1}, impl::data_type::f32);
    impl::logical_tensor_t bias
            = utils::logical_tensor_init(2, {1}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            3, {2, 1, 1}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src, weight, bias};
    std::vector<impl::logical_tensor_t> outputs {dst};

    auto &op_factory = get_dnnl_kernel_registry();
    auto matmul_kernel = op_factory.create_kernel(matmul_op);
    matmul_kernel->compile(&matmul_op, &engine, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src, src_data.data());
    impl::tensor_t weight_ts(weight, weight_data.data());
    impl::tensor_t bias_ts(bias, bias_data.data());
    impl::tensor_t dst_ts(outputs[0], dst_data.data());

    impl::stream_t &strm = get_stream();
    matmul_kernel->execute(
            &matmul_op, &strm, {src_ts, weight_ts, bias_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, matmul_bias_relu6_fusion) {
    impl::op_t matmul_op(impl::op_kind::matmul_bias_relu6);
    matmul_op.set_attr<float>("min", 0.0);
    matmul_op.set_attr<float>("max", 6.0);

    impl::engine_t &engine = get_engine();

    test::vector<float> src_data {-2.0, -1.5};
    test::vector<float> weight_data {-2.0, -1.5};
    test::vector<float> bias_data {1.0};
    test::vector<float> ref_dst_data {6.0};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {1, 2}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 1}, impl::data_type::f32);
    impl::logical_tensor_t bias
            = utils::logical_tensor_init(2, {1, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            3, {1, 1}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src, weight, bias};
    std::vector<impl::logical_tensor_t> outputs {dst};

    auto &op_factory = get_dnnl_kernel_registry();
    auto matmul_kernel = op_factory.create_kernel(matmul_op);
    matmul_kernel->compile(&matmul_op, &engine, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src, src_data.data());
    impl::tensor_t weight_ts(weight, weight_data.data());
    impl::tensor_t bias_ts(bias, bias_data.data());
    impl::tensor_t dst_ts(outputs[0], dst_data.data());

    impl::stream_t &strm = get_stream();
    matmul_kernel->execute(
            &matmul_op, &strm, {src_ts, weight_ts, bias_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, matmul_bias_hardtanh_fusion) {
    impl::op_t matmul_op(impl::op_kind::matmul_bias_hardtanh);
    matmul_op.set_attr<float>("min", -3.0);
    matmul_op.set_attr<float>("max", 3.0);

    impl::engine_t &engine = get_engine();

    test::vector<float> src_data {-2.0, -1.5};
    test::vector<float> weight_data {-2.0, -1.5};
    test::vector<float> bias_data {1.0};
    test::vector<float> ref_dst_data {3.0};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {1, 2}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 1}, impl::data_type::f32);
    impl::logical_tensor_t bias
            = utils::logical_tensor_init(2, {1, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            3, {1, 1}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src, weight, bias};
    std::vector<impl::logical_tensor_t> outputs {dst};

    auto &op_factory = get_dnnl_kernel_registry();
    auto matmul_kernel = op_factory.create_kernel(matmul_op);
    matmul_kernel->compile(&matmul_op, &engine, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src, src_data.data());
    impl::tensor_t weight_ts(weight, weight_data.data());
    impl::tensor_t bias_ts(bias, bias_data.data());
    impl::tensor_t dst_ts(outputs[0], dst_data.data());

    impl::stream_t &strm = get_stream();
    matmul_kernel->execute(
            &matmul_op, &strm, {src_ts, weight_ts, bias_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, matmul_bias_elu_fusion) {
    impl::op_t matmul_op(impl::op_kind::matmul_bias_elu);
    matmul_op.set_attr<float>("alpha", 1.f);

    impl::engine_t &engine = get_engine();

    test::vector<float> src_data {-2.0, -1.5};
    test::vector<float> weight_data {2.0, -1.5};
    test::vector<float> bias_data {1.0};
    test::vector<float> ref_dst_data {-0.75};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);
    ref_dst_data[0] = static_cast<float>(exp(-0.75) - 1);
    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {1, 2}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 1}, impl::data_type::f32);
    impl::logical_tensor_t bias
            = utils::logical_tensor_init(2, {1, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            3, {1, 1}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src, weight, bias};
    std::vector<impl::logical_tensor_t> outputs {dst};

    auto &op_factory = get_dnnl_kernel_registry();
    auto matmul_kernel = op_factory.create_kernel(matmul_op);
    matmul_kernel->compile(&matmul_op, &engine, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src, src_data.data());
    impl::tensor_t weight_ts(weight, weight_data.data());
    impl::tensor_t bias_ts(bias, bias_data.data());
    impl::tensor_t dst_ts(outputs[0], dst_data.data());

    impl::stream_t &strm = get_stream();
    matmul_kernel->execute(
            &matmul_op, &strm, {src_ts, weight_ts, bias_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, matmul_bias_sigmoid_fusion) {
    impl::op_t matmul_op(impl::op_kind::matmul_bias_sigmoid);
    impl::engine_t &engine = get_engine();

    test::vector<float> src_data {-2.0, -1.5};
    test::vector<float> weight_data {2.0, -1.5};
    test::vector<float> bias_data {1.0};
    test::vector<float> ref_dst_data {-0.75};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);
    ref_dst_data[0] = static_cast<float>(1 / (exp(-ref_dst_data[0]) + 1));
    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {1, 2}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 1}, impl::data_type::f32);
    impl::logical_tensor_t bias
            = utils::logical_tensor_init(2, {1, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            3, {1, 1}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src, weight, bias};
    std::vector<impl::logical_tensor_t> outputs {dst};

    auto &op_factory = get_dnnl_kernel_registry();
    auto matmul_kernel = op_factory.create_kernel(matmul_op);
    matmul_kernel->compile(&matmul_op, &engine, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src, src_data.data());
    impl::tensor_t weight_ts(weight, weight_data.data());
    impl::tensor_t bias_ts(bias, bias_data.data());
    impl::tensor_t dst_ts(outputs[0], dst_data.data());

    impl::stream_t &strm = get_stream();
    matmul_kernel->execute(
            &matmul_op, &strm, {src_ts, weight_ts, bias_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, matmul_bias_add_fusion) {
    impl::op_t matmul_op(impl::op_kind::matmul_bias_add);
    impl::engine_t &engine = get_engine();

    test::vector<float> src_data {-2.0, -1.5};
    test::vector<float> weight_data {2.0, -1.5};
    test::vector<float> bias_data {1.0};
    test::vector<float> post_src_data {-2.0};
    test::vector<float> ref_dst_data {-2.75};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {1, 2}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 1}, impl::data_type::f32);
    impl::logical_tensor_t bias
            = utils::logical_tensor_init(2, {1, 1}, impl::data_type::f32);
    impl::logical_tensor_t post_src
            = utils::logical_tensor_init(3, {1, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            4, {1, 1}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src, weight, bias, post_src};
    std::vector<impl::logical_tensor_t> outputs {dst};

    auto &op_factory = get_dnnl_kernel_registry();
    auto matmul_kernel = op_factory.create_kernel(matmul_op);
    matmul_kernel->compile(&matmul_op, &engine, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src, src_data.data());
    impl::tensor_t weight_ts(weight, weight_data.data());
    impl::tensor_t bias_ts(bias, bias_data.data());
    impl::tensor_t post_src_ts(post_src, post_src_data.data());
    impl::tensor_t dst_ts(outputs[0], dst_data.data());

    impl::stream_t &strm = get_stream();
    matmul_kernel->execute(&matmul_op, &strm,
            {src_ts, weight_ts, bias_ts, post_src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, matmul_bias_add_relu_fusion) {
    impl::op_t matmul_op(impl::op_kind::matmul_bias_add_relu);
    impl::engine_t &engine = get_engine();

    test::vector<float> src_data {-2.0, -1.5};
    test::vector<float> weight_data {2.0, -1.5};
    test::vector<float> bias_data {1.0};
    test::vector<float> post_src_data {-2.0};
    test::vector<float> ref_dst_data {0.0};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {1, 2}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 1}, impl::data_type::f32);
    impl::logical_tensor_t bias
            = utils::logical_tensor_init(2, {1, 1}, impl::data_type::f32);
    impl::logical_tensor_t post_src
            = utils::logical_tensor_init(3, {1, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            4, {1, 1}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src, weight, bias, post_src};
    std::vector<impl::logical_tensor_t> outputs {dst};

    auto &op_factory = get_dnnl_kernel_registry();
    auto matmul_kernel = op_factory.create_kernel(matmul_op);
    matmul_kernel->compile(&matmul_op, &engine, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src, src_data.data());
    impl::tensor_t weight_ts(weight, weight_data.data());
    impl::tensor_t bias_ts(bias, bias_data.data());
    impl::tensor_t post_src_ts(post_src, post_src_data.data());
    impl::tensor_t dst_ts(outputs[0], dst_data.data());

    impl::stream_t &strm = get_stream();
    matmul_kernel->execute(&matmul_op, &strm,
            {src_ts, weight_ts, bias_ts, post_src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, matmul_bias_bn) {
    impl::op_t matmul_op(impl::op_kind::matmul_bias_bn);
    matmul_op.set_attr<float>("epsilon", 0.f);
    matmul_op.set_attr<bool>("transpose_b", true);
    impl::engine_t &engine = get_engine();

    test::vector<float> src_data {-2.0, -1.5, -3.0, 0.5};
    test::vector<float> weight_data {2.0, -1.5, 1.0, -1.0};
    test::vector<float> bias_data {1.0, -0.5};
    test::vector<float> scale_data {2.0, 1.0};
    test::vector<float> shift_data {-1.5, 0.5};
    test::vector<float> mean_data {1.0, -2.0};
    test::vector<float> varience_data {1.0, 0.25};
    test::vector<float> ref_dst_data {-5.0, -3.5};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 1, 2}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 1, 2}, impl::data_type::f32);
    impl::logical_tensor_t bias
            = utils::logical_tensor_init(2, {2, 1, 1}, impl::data_type::f32);
    impl::logical_tensor_t scale
            = utils::logical_tensor_init(3, {2, 1, 1}, impl::data_type::f32);
    impl::logical_tensor_t shift
            = utils::logical_tensor_init(4, {2, 1, 1}, impl::data_type::f32);
    impl::logical_tensor_t mean
            = utils::logical_tensor_init(5, {2, 1, 1}, impl::data_type::f32);
    impl::logical_tensor_t varience
            = utils::logical_tensor_init(6, {2, 1, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            7, {2, 1, 1}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {
            src, weight, bias, scale, shift, mean, varience};
    std::vector<impl::logical_tensor_t> outputs {dst};

    auto &op_factory = get_dnnl_kernel_registry();
    auto matmul_kernel = op_factory.create_kernel(matmul_op);
    matmul_kernel->compile(&matmul_op, &engine, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src, src_data.data());
    impl::tensor_t weight_ts(weight, weight_data.data());
    impl::tensor_t bias_ts(bias, bias_data.data());
    impl::tensor_t scale_ts(scale, scale_data.data());
    impl::tensor_t shift_ts(shift, shift_data.data());
    impl::tensor_t mean_ts(mean, mean_data.data());
    impl::tensor_t varience_ts(varience, varience_data.data());
    impl::tensor_t dst_ts(outputs[0], dst_data.data());

    impl::stream_t &strm = get_stream();
    matmul_kernel->execute(&matmul_op, &strm,
            {src_ts, weight_ts, bias_ts, scale_ts, shift_ts, mean_ts,
                    varience_ts},
            {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, max_pool) {
    using dims = impl::dnnl_impl::dims;
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> ref_dst {-0.5, 2.0, 3.0, 4.0};
    test::vector<float> dst(ref_dst.size(), 0.0);

    impl::op_t max_pool_op(impl::op_kind::MaxPool);
    max_pool_op.set_attr<dims>("strides", {2, 2});
    max_pool_op.set_attr<dims>("kernel", {2, 2});
    max_pool_op.set_attr<dims>("pads_begin", {0, 0});
    max_pool_op.set_attr<dims>("pads_end", {0, 0});
    max_pool_op.set_attr<std::string>("data_format", "NCX");
    max_pool_op.set_attr<dims>("dilations", {1, 1});

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 1, 2, 2}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto max_pool_kernel = op_factory.create_kernel(max_pool_op);

    max_pool_kernel->compile(&max_pool_op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());

    impl::stream_t &strm = get_stream();
    max_pool_kernel->execute(&max_pool_op, &strm, {src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, avg_pool_exclude_pad) {
    using dims = impl::dnnl_impl::dims;
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            3.0, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> ref_dst {
            -2.0, 0.25, 0.5, 0.75, 0.5, 0.75, 3.0, -1.5, 4.0};
    test::vector<float> dst(ref_dst.size(), 0.0);

    impl::op_t avg_pool_op(impl::op_kind::AvgPool);
    avg_pool_op.set_attr<dims>("strides", {2, 2});
    avg_pool_op.set_attr<dims>("kernel", {2, 2});
    avg_pool_op.set_attr<dims>("pads_begin", {1, 1});
    avg_pool_op.set_attr<dims>("pads_end", {1, 1});
    avg_pool_op.set_attr<bool>("exclude_pad", true);
    avg_pool_op.set_attr<std::string>("data_format", "NCX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 1, 3, 3}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto avg_pool_kernel = op_factory.create_kernel(avg_pool_op);

    avg_pool_kernel->compile(&avg_pool_op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());

    impl::stream_t &strm = get_stream();
    avg_pool_kernel->execute(&avg_pool_op, &strm, {src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, avg_pool_include_pad) {
    using dims = impl::dnnl_impl::dims;
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            3.0, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> ref_dst {
            -0.5, 0.125, 0.125, 0.375, 0.5, 0.375, 0.75, -0.75, 1.0};
    test::vector<float> dst(ref_dst.size(), 0.0);

    impl::op_t avg_pool_op(impl::op_kind::AvgPool);
    avg_pool_op.set_attr<dims>("strides", {2, 2});
    avg_pool_op.set_attr<dims>("kernel", {2, 2});
    avg_pool_op.set_attr<dims>("pads_begin", {1, 1});
    avg_pool_op.set_attr<dims>("pads_end", {1, 1});
    avg_pool_op.set_attr<std::string>("data_format", "NCX");
    avg_pool_op.set_attr<bool>("exclude_pad", false);

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 1, 3, 3}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto avg_pool_kernel = op_factory.create_kernel(avg_pool_op);

    avg_pool_kernel->compile(&avg_pool_op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());

    impl::stream_t &strm = get_stream();
    avg_pool_kernel->execute(&avg_pool_op, &strm, {src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, Convolution_NCX_OIX) {
    using dims = impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> ref_dst {-1.0, 2.5, 5.0, 1.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    impl::op_t conv_op(impl::op_kind::Convolution);
    conv_op.set_attr<dims>("strides", dims {1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_op.set_attr<int64_t>("groups", 0);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 1, 2, 2}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto conv_kernel = op_factory.create_kernel(conv_op);

    conv_kernel->compile(&conv_op, &eng, inputs, outputs);
    ASSERT_EQ(dst_lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    impl::stream_t &strm = get_stream();
    conv_kernel->execute(&conv_op, &strm, {src_ts, weight_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, Convolution3D_NCX_OIX) {
    using dims = std::vector<int64_t>;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> ref_dst {-1.0, 2.5, 5.0, 1.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    impl::op_t conv_op(impl::op_kind::Convolution);
    conv_op.set_attr<dims>("strides", dims {1, 1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0, 0});
    conv_op.set_attr<int64_t>("groups", 0);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {1, 1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt = utils::logical_tensor_init(
            1, {1, 1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {1, 1, 1, 2, 2}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto conv_kernel = op_factory.create_kernel(conv_op);

    conv_kernel->compile(&conv_op, &eng, inputs, outputs);
    ASSERT_EQ(dst_lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    impl::stream_t &strm = get_stream();
    conv_kernel->execute(&conv_op, &strm, {src_ts, weight_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, Convolution_NCX_XIO) {
    using dims = impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> ref_dst {-1.0, 2.5, 5.0, 1.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    impl::op_t conv_op(impl::op_kind::Convolution);
    conv_op.set_attr<dims>("strides", dims {1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_op.set_attr<int64_t>("groups", 0);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "XIO");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {3, 3, 1, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 1, 2, 2}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto conv_kernel = op_factory.create_kernel(conv_op);

    conv_kernel->compile(&conv_op, &eng, inputs, outputs);
    ASSERT_EQ(dst_lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    impl::stream_t &strm = get_stream();
    conv_kernel->execute(&conv_op, &strm, {src_ts, weight_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, Convolution3D_NCX_XIO) {
    using dims = impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> ref_dst {-1.0, 2.5, 5.0, 1.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    impl::op_t conv_op(impl::op_kind::Convolution);
    conv_op.set_attr<dims>("strides", dims {1, 1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0, 0});
    conv_op.set_attr<int64_t>("groups", 0);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "XIO");

    // prepare logical tensor
    impl::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {1, 1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt = utils::logical_tensor_init(
            1, {1, 3, 3, 1, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {1, 1, 1, 2, 2}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto conv_kernel = op_factory.create_kernel(conv_op);

    conv_kernel->compile(&conv_op, &eng, inputs, outputs);
    ASSERT_EQ(dst_lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    impl::stream_t &strm = get_stream();
    conv_kernel->execute(&conv_op, &strm, {src_ts, weight_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, Convolution_NXC_XIO) {
    using dims = impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {
            -3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0, 2.5, -1.0, 0};
    test::vector<float> weight {
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0};
    test::vector<float> ref_dst {0.5};
    test::vector<float> dst {0.0};
    impl::op_t conv_op(impl::op_kind::Convolution);
    conv_op.set_attr<dims>("strides", dims {1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_op.set_attr<int64_t>("groups", 0);
    conv_op.set_attr<std::string>("data_format", "NXC");
    conv_op.set_attr<std::string>("filter_format", "XIO");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 3, 4, 1}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {3, 4, 1, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 1, 1, 1}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto conv_kernel = op_factory.create_kernel(conv_op);

    conv_kernel->compile(&conv_op, &eng, inputs, outputs);
    ASSERT_EQ(dst_lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    impl::stream_t &strm = get_stream();
    conv_kernel->execute(&conv_op, &strm, {src_ts, weight_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, Convolution3D_NXC_XIO) {
    using dims = impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {
            -3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0, 2.5, -1.0, 0};
    test::vector<float> weight {
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0};
    test::vector<float> ref_dst {0.5};
    test::vector<float> dst {0.0};
    impl::op_t conv_op(impl::op_kind::Convolution);
    conv_op.set_attr<dims>("strides", dims {1, 1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0, 0});
    conv_op.set_attr<int64_t>("groups", 0);
    conv_op.set_attr<std::string>("data_format", "NXC");
    conv_op.set_attr<std::string>("filter_format", "XIO");

    // prepare logical tensor
    impl::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {1, 1, 3, 4, 1}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt = utils::logical_tensor_init(
            1, {1, 3, 4, 1, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {1, 1, 1, 1, 1}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto conv_kernel = op_factory.create_kernel(conv_op);

    conv_kernel->compile(&conv_op, &eng, inputs, outputs);
    ASSERT_EQ(dst_lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    impl::stream_t &strm = get_stream();
    conv_kernel->execute(&conv_op, &strm, {src_ts, weight_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, Convolution_NXC_OIX) {
    using dims = impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> ref_dst {-1.0, 2.5, 5.0, 1.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    impl::op_t conv_op(impl::op_kind::Convolution);
    conv_op.set_attr<dims>("strides", dims {1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_op.set_attr<int64_t>("groups", 0);
    conv_op.set_attr<std::string>("data_format", "NXC");
    conv_op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 4, 4, 1}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 2, 2, 1}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto conv_kernel = op_factory.create_kernel(conv_op);

    conv_kernel->compile(&conv_op, &eng, inputs, outputs);
    ASSERT_EQ(dst_lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    impl::stream_t &strm = get_stream();
    conv_kernel->execute(&conv_op, &strm, {src_ts, weight_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, Convolution3D_NXC_OIX) {
    using dims = impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> ref_dst {-1.0, 2.5, 5.0, 1.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    impl::op_t conv_op(impl::op_kind::Convolution);
    conv_op.set_attr<dims>("strides", dims {1, 1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0, 0});
    conv_op.set_attr<int64_t>("groups", 0);
    conv_op.set_attr<std::string>("data_format", "NXC");
    conv_op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {1, 1, 4, 4, 1}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt = utils::logical_tensor_init(
            1, {1, 1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {1, 1, 2, 2, 1}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto conv_kernel = op_factory.create_kernel(conv_op);

    conv_kernel->compile(&conv_op, &eng, inputs, outputs);
    ASSERT_EQ(dst_lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    impl::stream_t &strm = get_stream();
    conv_kernel->execute(&conv_op, &strm, {src_ts, weight_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, ConvolutionF16F16F16) {
    using dims = impl::dnnl_impl::dims;

    impl::engine_t &eng = get_engine();
    SKIP_IF(eng.kind() != impl::engine_kind::gpu,
            "Skip fp16 test for non-GPU device.");
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> ref_dst {-1.0, 2.5, 5.0, 1.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    impl::op_t conv_op(impl::op_kind::Convolution);
    conv_op.set_attr<dims>("strides", dims {1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_op.set_attr<int64_t>("groups", 0);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f16);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f16);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 1, 2, 2}, impl::data_type::f16);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto conv_kernel = op_factory.create_kernel(conv_op);

    conv_kernel->compile(&conv_op, &eng, inputs, outputs);
    ASSERT_EQ(dst_lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    impl::stream_t &strm = get_stream();
    conv_kernel->execute(&conv_op, &strm, {src_ts, weight_ts}, {dst_ts});
    strm.wait();
}

TEST(operator_compile, ConvolutionBF16BF16BF16) {
    using dims = impl::dnnl_impl::dims;

    impl::engine_t &eng = get_engine();

    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF(isa < dnnl_cpu_isa_avx512_core
                    && eng.kind() == impl::engine_kind::cpu,
            "Skip bf16 examples for systems that do not support avx512_core.");

    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> ref_dst {-1.0, 2.5, 5.0, 1.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    impl::op_t conv_op(impl::op_kind::Convolution);
    conv_op.set_attr<dims>("strides", dims {1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_op.set_attr<int64_t>("groups", 0);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {1, 1, 4, 4}, impl::data_type::bf16);
    impl::logical_tensor_t weight_lt = utils::logical_tensor_init(
            1, {1, 1, 3, 3}, impl::data_type::bf16);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {1, 1, 2, 2}, impl::data_type::bf16);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto conv_kernel = op_factory.create_kernel(conv_op);

    conv_kernel->compile(&conv_op, &eng, inputs, outputs);
    ASSERT_EQ(dst_lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    impl::stream_t &strm = get_stream();
    conv_kernel->execute(&conv_op, &strm, {src_ts, weight_ts}, {dst_ts});
    strm.wait();
}

TEST(operator_compile, ConvolutionBF16BF16F32) {
    using dims = impl::dnnl_impl::dims;

    impl::engine_t &eng = get_engine();
    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF(isa < dnnl_cpu_isa_avx512_core
                    && eng.kind() == impl::engine_kind::cpu,
            "Skip bf16 examples for systems that do not support avx512_core.");

    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> ref_dst {-1.0, 2.5, 5.0, 1.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    impl::op_t conv_op(impl::op_kind::Convolution);
    conv_op.set_attr<dims>("strides", dims {1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_op.set_attr<int64_t>("groups", 0);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {1, 1, 4, 4}, impl::data_type::bf16);
    impl::logical_tensor_t weight_lt = utils::logical_tensor_init(
            1, {1, 1, 3, 3}, impl::data_type::bf16);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 1, 2, 2}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto conv_kernel = op_factory.create_kernel(conv_op);

    conv_kernel->compile(&conv_op, &eng, inputs, outputs);
    ASSERT_EQ(dst_lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    impl::stream_t &strm = get_stream();
    conv_kernel->execute(&conv_op, &strm, {src_ts, weight_ts}, {dst_ts});
    strm.wait();
}

TEST(operator_kernel, group_convolution) {
    using dims = impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();

    impl::op_t conv_op(impl::op_kind::Convolution);
    conv_op.set_attr<dims>("strides", dims {1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_op.set_attr<int64_t>("groups", 4);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t conv_src_lt = utils::logical_tensor_init(
            0, {8, 32, 16, 16}, impl::data_type::f32);
    impl::logical_tensor_t conv_weight_lt = utils::logical_tensor_init(
            1, {32, 8, 1, 1}, impl::data_type::f32);
    impl::logical_tensor_t conv_dst_lt = utils::logical_tensor_init(
            2, {8, 32, 16, 16}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> conv_inputs {
            conv_src_lt, conv_weight_lt};
    std::vector<impl::logical_tensor_t> conv_outputs {conv_dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto conv_kernel = op_factory.create_kernel(conv_op);

    conv_kernel->compile(&conv_op, &eng, conv_inputs, conv_outputs);
    ASSERT_EQ(conv_outputs[0].layout_type, impl::layout_type::strided);

    test::vector<float> conv_src(8 * 32 * 16 * 16, 1);
    test::vector<float> conv_weight(32 * 8 * 1 * 1, 1);
    test::vector<float> relu_dst(8 * 32 * 16 * 16, 1);
    size_t conv_dst_size
            = impl::dnnl_impl::dnnl_backend::get_singleton().get_mem_size(
                    conv_outputs[0]);
    test::vector<float> conv_dst(conv_dst_size / sizeof(float), 1);

    impl::tensor_t conv_src_ts(conv_src_lt, conv_src.data());
    impl::tensor_t conv_weight_ts(conv_weight_lt, conv_weight.data());
    impl::tensor_t conv_dst_ts(conv_outputs[0], conv_dst.data());

    impl::stream_t &strm = get_stream();
    conv_kernel->execute(
            &conv_op, &strm, {conv_src_ts, conv_weight_ts}, {conv_dst_ts});
    strm.wait();

    for (size_t i = 0; i < relu_dst.size(); ++i) {
        ASSERT_FLOAT_EQ(conv_dst[i], 8);
    }
}

TEST(operator_kernel, ConvolutionBackpropData) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;

    impl::engine_t &eng = get_engine();

    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> ref_diff_src {0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0,
            0.0, 3.0, 3.0, 1.0, 2.0, 3.0, 2.0, 3.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> diff_dst {0.0, 1.0, 2.0, 3.0};
    test::vector<float> diff_src(src.size(), 0.0);

    impl::op_t conv_op(impl::op_kind::ConvolutionBackpropData);
    conv_op.set_attr<dims>("strides", dims {1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_op.set_attr<int64_t>("groups", 0);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t diff_src_lt
            = utils::logical_tensor_init(1, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(2, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t diff_dst_lt
            = utils::logical_tensor_init(3, {1, 1, 2, 2}, impl::data_type::f32);

    auto &op_factory = get_dnnl_kernel_registry();
    auto conv_bwd_kernel = op_factory.create_kernel(conv_op);

    conv_bwd_kernel->compile(
            &conv_op, &eng, {diff_dst_lt, weight_lt}, {diff_src_lt});
    ASSERT_EQ(diff_src_lt.layout_type, impl::layout_type::strided);

    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t diff_dst_ts(diff_dst_lt, diff_dst.data());
    impl::tensor_t diff_src_ts(diff_src_lt, diff_src.data());

    impl::stream_t &strm = get_stream();
    conv_bwd_kernel->execute(
            &conv_op, &strm, {diff_dst_ts, weight_ts}, {diff_src_ts});
    strm.wait();
    for (size_t i = 0; i < diff_src.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_src[i], ref_diff_src[i]);
    }
}

TEST(operator_kernel, conv_bwd_weight_bias) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;

    impl::engine_t &eng = get_engine();

    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> diff_dst {0.0, 1.0, 2.0, 3.0};
    test::vector<float> ref_diff_weights {
            -5.5, 3.0, 7.0, 10.5, 3.0, -0.5, 2.5, -8.0, 10.0};
    test::vector<float> ref_diff_bias {6.0};
    test::vector<float> diff_weights(ref_diff_weights.size(), 0.0);
    test::vector<float> diff_bias(ref_diff_bias.size(), 0.0);

    impl::op_t conv_op(impl::op_kind::conv_bwd_f_biasadd_bwd);
    conv_op.set_attr<dims>("strides", dims {1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_op.set_attr<int64_t>("groups", 0);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "OIX");

    // prepare input/output logical_tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t diff_weights_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t diff_bias_lt
            = utils::logical_tensor_init(2, dims {1}, impl::data_type::f32);
    impl::logical_tensor_t diff_dst_lt
            = utils::logical_tensor_init(3, {1, 1, 2, 2}, impl::data_type::f32);

    auto &op_factory = get_dnnl_kernel_registry();
    auto conv_bwd_kernel = op_factory.create_kernel(conv_op);

    conv_bwd_kernel->compile(&conv_op, &eng, {src_lt, diff_dst_lt},
            {diff_weights_lt, diff_bias_lt});
    ASSERT_EQ(diff_weights_lt.layout_type, impl::layout_type::strided);
    ASSERT_EQ(diff_bias_lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t diff_dst_ts(diff_dst_lt, diff_dst.data());
    impl::tensor_t diff_weights_ts(diff_weights_lt, diff_weights.data());
    impl::tensor_t diff_bias_ts(diff_bias_lt, diff_bias.data());

    impl::stream_t &strm = get_stream();
    conv_bwd_kernel->execute(&conv_op, &strm, {src_ts, diff_dst_ts},
            {diff_weights_ts, diff_bias_ts});
    strm.wait();
    for (size_t i = 0; i < diff_weights.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_weights[i], ref_diff_weights[i]);
    }
    for (size_t i = 0; i < diff_bias.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_bias[i], ref_diff_bias[i]);
    }
}

TEST(operator_compile, conv_relu_unfused) {
    using dims = impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();

    impl::op_t conv_op(impl::op_kind::Convolution);
    conv_op.set_attr<dims>("strides", dims {1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_op.set_attr<int64_t>("groups", 0);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "OIX");

    impl::op_t relu_op(impl::op_kind::ReLU);

    // prepare logical tensor
    impl::logical_tensor_t conv_src_lt = utils::logical_tensor_init(
            0, {8, 32, 16, 16}, impl::data_type::f32);
    impl::logical_tensor_t conv_weight_lt = utils::logical_tensor_init(
            1, {32, 32, 1, 1}, impl::data_type::f32);
    impl::logical_tensor_t conv_dst_lt = utils::logical_tensor_init(
            2, {8, 32, 16, 16}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> conv_inputs {
            conv_src_lt, conv_weight_lt};
    std::vector<impl::logical_tensor_t> conv_outputs {conv_dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto conv_kernel = op_factory.create_kernel(conv_op);

    conv_kernel->compile(&conv_op, &eng, conv_inputs, conv_outputs);
    ASSERT_EQ(conv_outputs[0].layout_type, impl::layout_type::opaque);

    impl::logical_tensor_t relu_dst_lt = utils::logical_tensor_init(
            3, {8, 32, 16, 16}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> relu_inputs {conv_outputs[0]};
    std::vector<impl::logical_tensor_t> relu_outputs {relu_dst_lt};

    auto relu_kernel = op_factory.create_kernel(relu_op);

    relu_kernel->compile(&relu_op, &eng, relu_inputs, relu_outputs);
    ASSERT_EQ(relu_outputs[0].layout_type, impl::layout_type::strided);

    test::vector<float> conv_src(8 * 32 * 16 * 16, 1);
    test::vector<float> conv_weight(32 * 32 * 1 * 1, 1);
    test::vector<float> relu_dst(8 * 32 * 16 * 16, 1);
    size_t conv_dst_size
            = impl::dnnl_impl::dnnl_backend::get_singleton().get_mem_size(
                    conv_outputs[0]);
    test::vector<float> conv_dst(conv_dst_size / sizeof(float), 1);

    impl::tensor_t conv_src_ts(conv_src_lt, conv_src.data());
    impl::tensor_t conv_weight_ts(conv_weight_lt, conv_weight.data());
    impl::tensor_t conv_dst_ts(conv_outputs[0], conv_dst.data());

    impl::stream_t &strm = get_stream();
    conv_kernel->execute(
            &conv_op, &strm, {conv_src_ts, conv_weight_ts}, {conv_dst_ts});

    impl::tensor_t relu_src_ts(conv_outputs[0], conv_dst.data());
    impl::tensor_t relu_dst_ts(relu_outputs[0], relu_dst.data());

    relu_kernel->execute(&relu_op, &strm, {relu_src_ts}, {relu_dst_ts});
    strm.wait();

    for (size_t i = 0; i < relu_dst.size(); ++i) {
        ASSERT_FLOAT_EQ(relu_dst[i], 32);
    }
}

TEST(operator_compile, convolution_bn_fp32) {
    using dims = impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();

    impl::op_t conv_op(impl::op_kind::Convolution);
    conv_op.set_attr<dims>("strides", dims {1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_op.set_attr<int64_t>("groups", 0);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "OIX");

    impl::op_t bn_op(impl::op_kind::BatchNormInference);
    bn_op.set_attr<std::string>("data_format", "NCX");
    bn_op.set_attr("epsilon", 1e-6f);

    // prepare logical tensor
    impl::logical_tensor_t conv_src_lt = utils::logical_tensor_init(
            0, {8, 32, 16, 16}, impl::data_type::f32);
    impl::logical_tensor_t conv_weight_lt = utils::logical_tensor_init(
            1, {32, 32, 1, 1}, impl::data_type::f32);
    impl::logical_tensor_t conv_dst_lt = utils::logical_tensor_init(
            2, {8, 32, 16, 16}, impl::data_type::f32, impl::layout_type::any);
    std::vector<impl::logical_tensor_t> conv_inputs {
            conv_src_lt, conv_weight_lt};
    std::vector<impl::logical_tensor_t> conv_outputs {conv_dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto conv_kernel = op_factory.create_kernel(conv_op);
    conv_kernel->compile(&conv_op, &eng, conv_inputs, conv_outputs);

    impl::logical_tensor_t gamma
            = utils::logical_tensor_init(3, {32}, impl::data_type::f32);
    impl::logical_tensor_t beta
            = utils::logical_tensor_init(4, {32}, impl::data_type::f32);
    impl::logical_tensor_t scale
            = utils::logical_tensor_init(5, {32}, impl::data_type::f32);
    impl::logical_tensor_t shift
            = utils::logical_tensor_init(6, {32}, impl::data_type::f32);
    impl::logical_tensor_t bn_dst_lt = utils::logical_tensor_init(
            7, {8, 32, 16, 16}, impl::data_type::f32);
    std::vector<impl::logical_tensor_t> bn_inputs {
            conv_outputs[0], gamma, beta, scale, shift};
    std::vector<impl::logical_tensor_t> bn_outputs {bn_dst_lt};

    auto bn_kernel = op_factory.create_kernel(bn_op);
    bn_kernel->compile(&bn_op, &eng, bn_inputs, bn_outputs);

    test::vector<float> conv_src(8 * 32 * 16 * 16);
    test::vector<float> conv_weight(32 * 32 * 1 * 1);
    test::vector<float> bn_gamma(32);
    test::vector<float> bn_beta(32);
    test::vector<float> bn_scale(32);
    test::vector<float> bn_shift(32);

    // Initialize
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0f, 0.1f);

    std::generate(conv_src.begin(), conv_src.end(),
            [&]() { return distribution(generator); });
    std::generate(conv_weight.begin(), conv_weight.end(),
            [&]() { return distribution(generator); });
    std::generate(bn_gamma.begin(), bn_gamma.end(),
            [&]() { return distribution(generator); });
    std::generate(bn_beta.begin(), bn_beta.end(),
            [&]() { return distribution(generator); });
    std::generate(bn_scale.begin(), bn_scale.end(),
            [&]() { return distribution(generator); });
    std::generate(bn_shift.begin(), bn_shift.end(),
            [&]() { return distribution(generator); });

    auto &dnnl_backend = impl::dnnl_impl::dnnl_backend::get_singleton();
    size_t conv_dst_size = dnnl_backend.get_mem_size(conv_outputs[0]);
    test::vector<float> conv_dst(conv_dst_size / sizeof(float), 0.0);
    size_t bn_dst_size = dnnl_backend.get_mem_size(bn_outputs[0]);
    test::vector<float> bn_dst(bn_dst_size / sizeof(float), 0.0);

    impl::tensor_t conv_src_ts(conv_src_lt, conv_src.data());
    impl::tensor_t conv_weight_ts(conv_weight_lt, conv_weight.data());
    impl::tensor_t conv_dst_ts(conv_outputs[0], conv_dst.data());

    impl::stream_t &strm = get_stream();
    conv_kernel->execute(
            &conv_op, &strm, {conv_src_ts, conv_weight_ts}, {conv_dst_ts});

    impl::tensor_t bn_src_ts(conv_outputs[0], conv_dst.data());
    impl::tensor_t bn_gamma_ts(gamma, bn_gamma.data());
    impl::tensor_t bn_beta_ts(beta, bn_beta.data());
    impl::tensor_t bn_scale_ts(scale, bn_scale.data());
    impl::tensor_t bn_shift_ts(shift, bn_shift.data());
    impl::tensor_t bn_dst_ts(bn_outputs[0], bn_dst.data());

    bn_kernel->execute(&bn_op, &strm,
            {bn_src_ts, bn_gamma_ts, bn_beta_ts, bn_scale_ts, bn_shift_ts},
            {bn_dst_ts});

    impl::op_t convbn_op(impl::op_kind::conv_bn);
    convbn_op.set_attr<dims>("strides", dims {1, 1});
    convbn_op.set_attr<dims>("dilations", dims {1, 1});
    convbn_op.set_attr<dims>("pads_begin", dims {0, 0});
    convbn_op.set_attr<dims>("pads_end", dims {0, 0});
    convbn_op.set_attr<int64_t>("groups", 0);
    convbn_op.set_attr<std::string>("data_format", "NCX");
    convbn_op.set_attr<std::string>("filter_format", "OIX");
    convbn_op.set_attr("epsilon", 1e-6f);
    impl::logical_tensor_t conv_bn_dst_lt = utils::logical_tensor_init(
            8, {8, 32, 16, 16}, impl::data_type::f32);
    std::vector<impl::logical_tensor_t> convbn_inputs {
            conv_src_lt, conv_weight_lt, gamma, beta, scale, shift};
    std::vector<impl::logical_tensor_t> convbn_outputs {conv_bn_dst_lt};
    auto convbn_kernel = op_factory.create_kernel(convbn_op);
    convbn_kernel->compile(&convbn_op, &eng, convbn_inputs, convbn_outputs);

    size_t convbn_dst_size = dnnl_backend.get_mem_size(convbn_outputs[0]);
    test::vector<float> convbn_dst(convbn_dst_size / sizeof(float), 0.0);
    impl::tensor_t convbn_dst_ts(convbn_outputs[0], convbn_dst.data());

    convbn_kernel->execute(&convbn_op, &strm,
            {conv_src_ts, conv_weight_ts, bn_gamma_ts, bn_beta_ts, bn_scale_ts,
                    bn_shift_ts},
            {convbn_dst_ts});
    strm.wait();

    float max_diff = 0;
    for (size_t i = 0; i < bn_dst.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(bn_dst[i] - convbn_dst[i]));
    }
    ASSERT_LT(max_diff, 1e-6f);
}

TEST(operator_kernel, conv_add) {
    using dims = impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> post_src {1.0, 2.0, 3.0, 4.0};
    test::vector<float> ref_dst {0.0, 4.5, 8.0, 5.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    impl::op_t conv_add_op(impl::op_kind::conv_add);
    conv_add_op.set_attr<dims>("strides", dims {1, 1});
    conv_add_op.set_attr<dims>("dilations", dims {1, 1});
    conv_add_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_add_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_add_op.set_attr<int64_t>("groups", 1);
    conv_add_op.set_attr<std::string>("data_format", "NCX");
    conv_add_op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t post_src_lt
            = utils::logical_tensor_init(2, {1, 1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 1, 2, 2}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt, post_src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto conv_add_kernel = op_factory.create_kernel(conv_add_op);

    conv_add_kernel->compile(&conv_add_op, &eng, inputs, outputs);
    ASSERT_EQ(dst_lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t post_src_ts(post_src_lt, post_src.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    impl::stream_t &strm = get_stream();
    conv_add_kernel->execute(
            &conv_add_op, &strm, {src_ts, weight_ts, post_src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, conv_add_relu) {
    using dims = impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> post_src {-1.0, -2.0, -3.0, -4.0};
    test::vector<float> ref_dst {0.0, 0.5, 2.0, 0.0};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    impl::op_t conv_add_relu_op(impl::op_kind::conv_add_relu);
    conv_add_relu_op.set_attr<dims>("strides", dims {1, 1});
    conv_add_relu_op.set_attr<dims>("dilations", dims {1, 1});
    conv_add_relu_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_add_relu_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_add_relu_op.set_attr<int64_t>("groups", 1);
    conv_add_relu_op.set_attr<std::string>("data_format", "NCX");
    conv_add_relu_op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t post_lt
            = utils::logical_tensor_init(2, {1, 1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 1, 2, 2}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt, post_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto conv_add_relu_kernel = op_factory.create_kernel(conv_add_relu_op);

    conv_add_relu_kernel->compile(&conv_add_relu_op, &eng, inputs, outputs);
    ASSERT_EQ(dst_lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t post_src_ts(dst_lt, post_src.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    impl::stream_t &strm = get_stream();
    conv_add_relu_kernel->execute(&conv_add_relu_op, &strm,
            {src_ts, weight_ts, post_src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, abs) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 3.5};
    test::vector<float> ref_dst {2.0, 1.5, 1.0, 0.5, 0.0, 3.5};
    test::vector<float> dst(src.size(), 0.0);

    impl::op_t op(impl::op_kind::Abs);

    auto &op_factory = get_dnnl_kernel_registry();
    auto this_kernel = op_factory.create_kernel(op);
    ASSERT_TRUE(this_kernel);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 2, 3}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    // compile the relu operator
    this_kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());

    impl::stream_t &strm = get_stream();
    this_kernel->execute(&op, &strm, {src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, elu) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 3.5};
    test::vector<float> ref_dst;
    test::vector<float> dst(src.size(), 0.0);

    impl::op_t op(impl::op_kind::Elu);
    op.set_attr<float>("alpha", 1.f);
    auto &op_factory = get_dnnl_kernel_registry();
    auto this_kernel = op_factory.create_kernel(op);
    ASSERT_TRUE(this_kernel);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 2, 3}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    // compile the relu operator
    this_kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());

    impl::stream_t &strm = get_stream();
    this_kernel->execute(&op, &strm, {src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src.size(); ++i) {
        float temp;
        if (src[i] > 0) {
            temp = src[i];
        } else {
            temp = static_cast<float>(exp(src[i]) - 1);
        }
        ref_dst.push_back(temp);
    }

    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, exp) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 3.5};
    test::vector<float> ref_dst;
    test::vector<float> dst(src.size(), 0.0);

    impl::op_t op(impl::op_kind::Exp);

    auto &op_factory = get_dnnl_kernel_registry();
    auto this_kernel = op_factory.create_kernel(op);
    ASSERT_TRUE(this_kernel);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 2, 3}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    // compile the relu operator
    this_kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());

    impl::stream_t &strm = get_stream();
    this_kernel->execute(&op, &strm, {src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src.size(); ++i) {
        float temp = static_cast<float>(exp(src[i]));
        ref_dst.push_back(temp);
    }

    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, hardtanh) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 3.5};
    test::vector<float> ref_dst {-1.0, -1.0, -1.0, -0.5, 0.0, 2.0};
    test::vector<float> dst(src.size(), 0.0);

    impl::op_t op(impl::op_kind::HardTanh);
    op.set_attr<float>("max", 2.f);
    op.set_attr<float>("min", -1.f);
    auto &op_factory = get_dnnl_kernel_registry();
    auto this_kernel = op_factory.create_kernel(op);
    ASSERT_TRUE(this_kernel);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 2, 3}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    // compile the relu operator
    this_kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());

    impl::stream_t &strm = get_stream();
    this_kernel->execute(&op, &strm, {src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, sqrt) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {2.0, 1.5, 1.0, 0.5, 0.0, 3.5};
    test::vector<float> ref_dst;
    test::vector<float> dst(src.size(), 0.0);

    impl::op_t op(impl::op_kind::Sqrt);

    auto &op_factory = get_dnnl_kernel_registry();
    auto this_kernel = op_factory.create_kernel(op);
    ASSERT_TRUE(this_kernel);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 2, 3}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    // compile the relu operator
    this_kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());

    impl::stream_t &strm = get_stream();
    this_kernel->execute(&op, &strm, {src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src.size(); ++i) {
        float temp = static_cast<float>(sqrt(src[i]));
        ref_dst.push_back(temp);
    }

    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, square) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 3.5};
    test::vector<float> ref_dst;
    test::vector<float> dst(src.size(), 0.0);

    impl::op_t op(impl::op_kind::Square);

    auto &op_factory = get_dnnl_kernel_registry();
    auto this_kernel = op_factory.create_kernel(op);
    ASSERT_TRUE(this_kernel);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 2, 3}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    // compile the relu operator
    this_kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());

    impl::stream_t &strm = get_stream();
    this_kernel->execute(&op, &strm, {src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src.size(); ++i) {
        float temp = src[i] * src[i];
        ref_dst.push_back(temp);
    }

    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, pow) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 3.5};
    test::vector<float> ref_dst;
    test::vector<float> dst(src.size(), 0.0);

    impl::op_t op(impl::op_kind::Pow);
    op.set_attr<float>("alpha", 1.f);
    op.set_attr<float>("beta", 3.f);
    auto &op_factory = get_dnnl_kernel_registry();
    auto this_kernel = op_factory.create_kernel(op);
    ASSERT_TRUE(this_kernel);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 2, 3}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    // compile the relu operator
    this_kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());

    impl::stream_t &strm = get_stream();
    this_kernel->execute(&op, &strm, {src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src.size(); ++i) {
        float temp = static_cast<float>(pow(src[i], 3.0));
        ref_dst.push_back(temp);
    }

    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, log) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {2.f, 1.5f, 1.f, 0.5f, 0.8f, 3.5f};
    test::vector<float> ref_dst;
    test::vector<float> dst(src.size(), 0.0);

    impl::op_t op(impl::op_kind::Log);

    auto &op_factory = get_dnnl_kernel_registry();
    auto this_kernel = op_factory.create_kernel(op);
    ASSERT_TRUE(this_kernel);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 2, 3}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    // compile the relu operator
    this_kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());

    impl::stream_t &strm = get_stream();
    this_kernel->execute(&op, &strm, {src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src.size(); ++i) {
        float temp = static_cast<float>(log(src[i]));
        ref_dst.push_back(temp);
    }

    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_TRUE(std::fabs(dst[i] - ref_dst[i]) < 0.00001);
    }
}

TEST(operator_kernel, tanh) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 3.5};
    test::vector<float> ref_dst;
    test::vector<float> dst(src.size(), 0.0);

    impl::op_t op(impl::op_kind::Tanh);

    auto &op_factory = get_dnnl_kernel_registry();
    auto this_kernel = op_factory.create_kernel(op);
    ASSERT_TRUE(this_kernel);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 2, 3}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    // compile the relu operator
    this_kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());

    impl::stream_t &strm = get_stream();
    this_kernel->execute(&op, &strm, {src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src.size(); ++i) {
        float temp = static_cast<float>(
                (exp(src[i]) - exp(-src[i])) / (exp(src[i]) + exp(-src[i])));
        ref_dst.push_back(temp);
    }

    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, conv_bias_abs) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0.0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> bias {-1.0};
    test::vector<float> ref_dst {2.0, 1.5, 4.0, 0.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};

    impl::op_t op(impl::op_kind::conv_bias_abs);

    op.set_attr<dims>("strides", {1, 1});
    op.set_attr<dims>("dilations", {1, 1});
    op.set_attr<dims>("pads_begin", {0, 0});
    op.set_attr<dims>("pads_end", {0, 0});
    op.set_attr<int64_t>("groups", 1);
    op.set_attr<std::string>("data_format", "NCX");
    op.set_attr<std::string>("filter_format", "OIX");
    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t bias_lt
            = utils::logical_tensor_init(2, dims {1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(4, {1, 1, 2, 2}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt, bias_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto kernel = op_factory.create_kernel(op);
    kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t bias_ts(bias_lt, bias.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());
    impl::stream_t &strm = get_stream();
    kernel->execute(&op, &strm, {src_ts, weight_ts, bias_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, conv_bias_elu) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0.0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> bias {-1.0};
    test::vector<float> ref_dst {-2.0, 1.5, 4.0, 0.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    ref_dst[0] = static_cast<float>(exp(-2) - 1);
    impl::op_t op(impl::op_kind::conv_bias_elu);

    op.set_attr<dims>("strides", {1, 1});
    op.set_attr<dims>("dilations", {1, 1});
    op.set_attr<dims>("pads_begin", {0, 0});
    op.set_attr<dims>("pads_end", {0, 0});
    op.set_attr<int64_t>("groups", 1);
    op.set_attr<float>("alpha", 1.f);
    op.set_attr<std::string>("data_format", "NCX");
    op.set_attr<std::string>("filter_format", "OIX");
    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t bias_lt
            = utils::logical_tensor_init(2, dims {1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(4, {1, 1, 2, 2}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt, bias_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto kernel = op_factory.create_kernel(op);
    kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t bias_ts(bias_lt, bias.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());
    impl::stream_t &strm = get_stream();
    kernel->execute(&op, &strm, {src_ts, weight_ts, bias_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, conv_bias_hardtanh) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0.0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> bias {-1.0};
    test::vector<float> ref_dst {0.0, 1.5, 3.0, 0.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    impl::op_t op(impl::op_kind::conv_bias_hardtanh);

    op.set_attr<dims>("strides", {1, 1});
    op.set_attr<dims>("dilations", {1, 1});
    op.set_attr<dims>("pads_begin", {0, 0});
    op.set_attr<dims>("pads_end", {0, 0});
    op.set_attr<int64_t>("groups", 1);
    op.set_attr<float>("min", 0.f);
    op.set_attr<float>("max", 3.f);
    op.set_attr<std::string>("data_format", "NCX");
    op.set_attr<std::string>("filter_format", "OIX");
    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t bias_lt
            = utils::logical_tensor_init(2, dims {1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(4, {1, 1, 2, 2}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt, bias_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto kernel = op_factory.create_kernel(op);
    kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t bias_ts(bias_lt, bias.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());
    impl::stream_t &strm = get_stream();
    kernel->execute(&op, &strm, {src_ts, weight_ts, bias_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, conv_bias_relu6) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0.0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> bias {2.0};
    test::vector<float> ref_dst {1.0, 4.5, 6.0, 3.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    impl::op_t op(impl::op_kind::conv_bias_relu6);

    op.set_attr<dims>("strides", {1, 1});
    op.set_attr<dims>("dilations", {1, 1});
    op.set_attr<dims>("pads_begin", {0, 0});
    op.set_attr<dims>("pads_end", {0, 0});
    op.set_attr<int64_t>("groups", 1);
    op.set_attr<float>("min", 0.f);
    op.set_attr<float>("max", 6.f);
    op.set_attr<std::string>("data_format", "NCX");
    op.set_attr<std::string>("filter_format", "OIX");
    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t bias_lt
            = utils::logical_tensor_init(2, dims {1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(4, {1, 1, 2, 2}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt, bias_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto kernel = op_factory.create_kernel(op);
    kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t bias_ts(bias_lt, bias.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());
    impl::stream_t &strm = get_stream();
    kernel->execute(&op, &strm, {src_ts, weight_ts, bias_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, conv_bias_sigmoid) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0.0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> bias {-1.0};
    test::vector<float> ref_dst {-2.0, 1.5, 4.0, 0.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    for (auto &rdst : ref_dst) {
        rdst = static_cast<float>(1 / (exp(-rdst) + 1));
    }
    impl::op_t op(impl::op_kind::conv_bias_sigmoid);

    op.set_attr<dims>("strides", {1, 1});
    op.set_attr<dims>("dilations", {1, 1});
    op.set_attr<dims>("pads_begin", {0, 0});
    op.set_attr<dims>("pads_end", {0, 0});
    op.set_attr<int64_t>("groups", 1);
    op.set_attr<std::string>("data_format", "NCX");
    op.set_attr<std::string>("filter_format", "OIX");
    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t bias_lt
            = utils::logical_tensor_init(2, dims {1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(4, {1, 1, 2, 2}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt, bias_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto kernel = op_factory.create_kernel(op);
    kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t bias_ts(bias_lt, bias.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());
    impl::stream_t &strm = get_stream();
    kernel->execute(&op, &strm, {src_ts, weight_ts, bias_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, conv_bias_sqrt) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0.0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> bias {1.0};
    test::vector<float> ref_dst {0.0, 3.5, 6.0, 2.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    for (auto &rdst : ref_dst) {
        rdst = static_cast<float>(sqrt(rdst));
    }
    impl::op_t op(impl::op_kind::conv_bias_sqrt);

    op.set_attr<dims>("strides", {1, 1});
    op.set_attr<dims>("dilations", {1, 1});
    op.set_attr<dims>("pads_begin", {0, 0});
    op.set_attr<dims>("pads_end", {0, 0});
    op.set_attr<int64_t>("groups", 1);

    op.set_attr<std::string>("data_format", "NCX");
    op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t bias_lt
            = utils::logical_tensor_init(2, dims {1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(4, {1, 1, 2, 2}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt, bias_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto kernel = op_factory.create_kernel(op);
    kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t bias_ts(bias_lt, bias.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());
    impl::stream_t &strm = get_stream();
    kernel->execute(&op, &strm, {src_ts, weight_ts, bias_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, conv_bias_square) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0.0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> bias {-1.0};
    test::vector<float> ref_dst {4.0, 2.25, 16.0, 0.25};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};

    impl::op_t op(impl::op_kind::conv_bias_square);

    op.set_attr<dims>("strides", {1, 1});
    op.set_attr<dims>("dilations", {1, 1});
    op.set_attr<dims>("pads_begin", {0, 0});
    op.set_attr<dims>("pads_end", {0, 0});
    op.set_attr<int64_t>("groups", 1);

    op.set_attr<std::string>("data_format", "NCX");
    op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t bias_lt
            = utils::logical_tensor_init(2, dims {1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(4, {1, 1, 2, 2}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt, bias_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto kernel = op_factory.create_kernel(op);
    kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t bias_ts(bias_lt, bias.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());
    impl::stream_t &strm = get_stream();
    kernel->execute(&op, &strm, {src_ts, weight_ts, bias_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, conv_bias_tanh) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0.0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> bias {-1.0};
    test::vector<float> ref_dst {-2.0, 1.5, 4.0, 0.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    for (auto &rdst : ref_dst) {
        rdst = static_cast<float>(
                (exp(rdst) - exp(-rdst)) / (exp(rdst) + exp(-rdst)));
    }
    impl::op_t op(impl::op_kind::conv_bias_tanh);

    op.set_attr<dims>("strides", {1, 1});
    op.set_attr<dims>("dilations", {1, 1});
    op.set_attr<dims>("pads_begin", {0, 0});
    op.set_attr<dims>("pads_end", {0, 0});
    op.set_attr<int64_t>("groups", 1);

    op.set_attr<std::string>("data_format", "NCX");
    op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t bias_lt
            = utils::logical_tensor_init(2, dims {1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(4, {1, 1, 2, 2}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt, bias_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto kernel = op_factory.create_kernel(op);
    kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t bias_ts(bias_lt, bias.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());
    impl::stream_t &strm = get_stream();
    kernel->execute(&op, &strm, {src_ts, weight_ts, bias_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, conv_bias_add_elu) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0.0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> post_src {-2.0, 1.0, -1.0, 0.0};
    test::vector<float> bias {-1.0};
    test::vector<float> ref_dst {-4.0, 2.5, 3.0, 0.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    ref_dst[0] = static_cast<float>(exp(-4) - 1);
    impl::op_t op(impl::op_kind::conv_bias_add_elu);

    op.set_attr<dims>("strides", {1, 1});
    op.set_attr<dims>("dilations", {1, 1});
    op.set_attr<dims>("pads_begin", {0, 0});
    op.set_attr<dims>("pads_end", {0, 0});
    op.set_attr<int64_t>("groups", 1);

    op.set_attr<float>("alpha", 1.f);
    op.set_attr<std::string>("data_format", "NCX");
    op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t bias_lt
            = utils::logical_tensor_init(2, dims {1}, impl::data_type::f32);
    impl::logical_tensor_t post_lt
            = utils::logical_tensor_init(3, {1, 1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(4, {1, 1, 2, 2}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {
            src_lt, weight_lt, bias_lt, post_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto kernel = op_factory.create_kernel(op);
    kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t bias_ts(bias_lt, bias.data());
    impl::tensor_t post_src_ts(outputs[0], post_src.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());
    impl::stream_t &strm = get_stream();
    kernel->execute(
            &op, &strm, {src_ts, weight_ts, bias_ts, post_src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, conv_bias_add_relu6) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0.0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> post_src {-2.0, 1.0, -1.0, 0.0};
    test::vector<float> bias {3.0};
    test::vector<float> ref_dst {0.0, 6.f, 6.f, 4.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};

    impl::op_t op(impl::op_kind::conv_bias_add_relu6);

    op.set_attr<dims>("strides", {1, 1});
    op.set_attr<dims>("dilations", {1, 1});
    op.set_attr<dims>("pads_begin", {0, 0});
    op.set_attr<dims>("pads_end", {0, 0});
    op.set_attr<int64_t>("groups", 1);

    op.set_attr<float>("min", 0.f);
    op.set_attr<float>("max", 6.f);
    op.set_attr<std::string>("data_format", "NCX");
    op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t bias_lt
            = utils::logical_tensor_init(2, dims {1}, impl::data_type::f32);
    impl::logical_tensor_t post_lt
            = utils::logical_tensor_init(3, {1, 1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(4, {1, 1, 2, 2}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {
            src_lt, weight_lt, bias_lt, post_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto kernel = op_factory.create_kernel(op);
    kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t bias_ts(bias_lt, bias.data());
    impl::tensor_t post_src_ts(outputs[0], post_src.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());
    impl::stream_t &strm = get_stream();
    kernel->execute(
            &op, &strm, {src_ts, weight_ts, bias_ts, post_src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, conv_add_elu) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0.0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> post_src {-2.0, 1.0, -1.0, 0.0};
    test::vector<float> ref_dst {-3.0, 3.5, 4.0, 1.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    for (auto &rdst : ref_dst) {
        rdst = rdst > 0 ? rdst : static_cast<float>((exp(rdst) - 1));
    }

    impl::op_t op(impl::op_kind::conv_add_elu);

    op.set_attr<dims>("strides", {1, 1});
    op.set_attr<dims>("dilations", {1, 1});
    op.set_attr<dims>("pads_begin", {0, 0});
    op.set_attr<dims>("pads_end", {0, 0});
    op.set_attr<int64_t>("groups", 1);

    op.set_attr<float>("alpha", 1.f);
    op.set_attr<std::string>("data_format", "NCX");
    op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t post_lt
            = utils::logical_tensor_init(2, {1, 1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 1, 2, 2}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt, post_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto kernel = op_factory.create_kernel(op);
    kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t post_src_ts(outputs[0], post_src.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());
    impl::stream_t &strm = get_stream();
    kernel->execute(&op, &strm, {src_ts, weight_ts, post_src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, conv_add_relu6) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0.0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> post_src {-2.0, 1.0, -1.0, 0.0};
    test::vector<float> ref_dst {0.0, 3.5, 4.f, 1.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};

    impl::op_t op(impl::op_kind::conv_add_relu6);

    op.set_attr<dims>("strides", {1, 1});
    op.set_attr<dims>("dilations", {1, 1});
    op.set_attr<dims>("pads_begin", {0, 0});
    op.set_attr<dims>("pads_end", {0, 0});
    op.set_attr<int64_t>("groups", 1);

    op.set_attr<float>("min", 0.f);
    op.set_attr<float>("max", 6.f);
    op.set_attr<std::string>("data_format", "NCX");
    op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t post_lt
            = utils::logical_tensor_init(2, {1, 1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 1, 2, 2}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt, post_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto kernel = op_factory.create_kernel(op);
    kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t post_src_ts(outputs[0], post_src.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());
    impl::stream_t &strm = get_stream();
    kernel->execute(&op, &strm, {src_ts, weight_ts, post_src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, softmax) {
    impl::op_t softmax_op(impl::op_kind::SoftMax);
    impl::engine_t &eng = get_engine();

    softmax_op.set_attr<int64_t>("axis", 0);

    test::vector<float> src_data {0.0, 1.0, 2.0, 0.0, 1.0, 2.0};
    test::vector<float> ref_dst_data {0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            1, {2, 3}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src};
    std::vector<impl::logical_tensor_t> outputs {dst};

    auto &op_factory = get_dnnl_kernel_registry();
    auto softmax_kernel = op_factory.create_kernel(softmax_op);

    softmax_kernel->compile(&softmax_op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src, src_data.data());
    impl::tensor_t dst_ts(outputs[0], dst_data.data());

    impl::stream_t &strm = get_stream();
    softmax_kernel->execute(&softmax_op, &strm, {src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, softmax_with_last_dim) {
    impl::op_t softmax_op(impl::op_kind::SoftMax);
    impl::engine_t &eng = get_engine();

    softmax_op.set_attr<int64_t>("axis", -1);

    test::vector<float> src_data {3.0, 3.0, 1.0, 1.0};
    test::vector<float> ref_dst_data {0.5, 0.5, 0.5, 0.5};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            1, {2, 2}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src};
    std::vector<impl::logical_tensor_t> outputs {dst};

    auto &op_factory = get_dnnl_kernel_registry();
    auto softmax_kernel = op_factory.create_kernel(softmax_op);

    softmax_kernel->compile(&softmax_op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src, src_data.data());
    impl::tensor_t dst_ts(outputs[0], dst_data.data());

    impl::stream_t &strm = get_stream();
    softmax_kernel->execute(&softmax_op, &strm, {src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, softmax_backward) {
    impl::op_t softmax_op(impl::op_kind::SoftMaxBackprop);
    impl::engine_t &eng = get_engine();

    softmax_op.set_attr<int64_t>("axis", 1);

    test::vector<float> dst {0.5, 0.5, 0.5, 0.5};
    test::vector<float> diff_dst {1.0, 5.0, 2.0, 4.0};
    test::vector<float> ref_diff_src {-1, 1, -0.5, 0.5};
    test::vector<float> diff_src(ref_diff_src.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(1, {2, 2}, impl::data_type::f32);
    impl::logical_tensor_t diff_dst_lt
            = utils::logical_tensor_init(2, {2, 2}, impl::data_type::f32);
    impl::logical_tensor_t diff_src_lt = utils::logical_tensor_init(
            3, {2, 2}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {diff_dst_lt, dst_lt};
    std::vector<impl::logical_tensor_t> outputs {diff_src_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto softmax_kernel = op_factory.create_kernel(softmax_op);

    softmax_kernel->compile(&softmax_op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t dst_ts(dst_lt, dst.data());
    impl::tensor_t diff_dst_ts(diff_dst_lt, diff_dst.data());
    impl::tensor_t diff_src_ts(outputs[0], diff_src.data());

    impl::stream_t &strm = get_stream();
    softmax_kernel->execute(
            &softmax_op, &strm, {diff_dst_ts, dst_ts}, {diff_src_ts});
    strm.wait();
    for (size_t i = 0; i < ref_diff_src.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_src[i], ref_diff_src[i]);
    }
}

TEST(operator_kernel, logsoftmax) {
    impl::op_t logsoftmax_op(impl::op_kind::LogSoftmax);
    impl::engine_t &eng = get_engine();

    logsoftmax_op.set_attr<int64_t>("axis", 0);

    test::vector<float> src_data {0.0, 1.0, 2.0, 0.0, 1.0, 2.0};
    test::vector<float> ref_dst_data {-0.6931472, -0.6931472, -0.6931472,
            -0.6931472, -0.6931472, -0.6931472};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            1, {2, 3}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src};
    std::vector<impl::logical_tensor_t> outputs {dst};

    auto &op_factory = get_dnnl_kernel_registry();
    auto softmax_kernel = op_factory.create_kernel(logsoftmax_op);

    softmax_kernel->compile(&logsoftmax_op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src, src_data.data());
    impl::tensor_t dst_ts(outputs[0], dst_data.data());

    impl::stream_t &strm = get_stream();
    softmax_kernel->execute(&logsoftmax_op, &strm, {src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, logsoftmax_backward) {
    impl::op_t logsoftmax_op(impl::op_kind::LogSoftmaxBackprop);
    impl::engine_t &eng = get_engine();

    logsoftmax_op.set_attr<int64_t>("axis", 1);

    test::vector<float> dst {-0.6931472, -0.6931472, -0.6931472, -0.6931472};
    test::vector<float> diff_dst {1.0, 5.0, 2.0, 4.0};
    test::vector<float> ref_diff_src {-2, 2, -1, 1};
    test::vector<float> diff_src(ref_diff_src.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(1, {2, 2}, impl::data_type::f32);
    impl::logical_tensor_t diff_dst_lt
            = utils::logical_tensor_init(2, {2, 2}, impl::data_type::f32);
    impl::logical_tensor_t diff_src_lt = utils::logical_tensor_init(
            3, {2, 2}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {diff_dst_lt, dst_lt};
    std::vector<impl::logical_tensor_t> outputs {diff_src_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto softmax_kernel = op_factory.create_kernel(logsoftmax_op);

    softmax_kernel->compile(&logsoftmax_op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t dst_ts(dst_lt, dst.data());
    impl::tensor_t diff_dst_ts(diff_dst_lt, diff_dst.data());
    impl::tensor_t diff_src_ts(outputs[0], diff_src.data());

    impl::stream_t &strm = get_stream();
    softmax_kernel->execute(
            &logsoftmax_op, &strm, {diff_dst_ts, dst_ts}, {diff_src_ts});
    strm.wait();
    for (size_t i = 0; i < ref_diff_src.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_src[i], ref_diff_src[i]);
    }
}

TEST(operator_kernel, avg_pool_backward_exclude_pad) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            3.0, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> diff_src(src.size(), 0.0);
    test::vector<float> ref_diff_src {-1.0, 1.5, 1.5, 10.0, 2.0, 4.0, 4.0, 4.0,
            2.0, 4.0, 4.0, 4.0, 12.0, -2.5, -2.5, -3.0};
    test::vector<float> diff_dst {
            -1.0, 3.0, 10.0, 4.0, 16.0, 8.0, 12.0, -5.0, -3.0};

    impl::op_t avg_pool_bwd_op(impl::op_kind::AvgPoolBackprop);
    avg_pool_bwd_op.set_attr<dims>("strides", {2, 2});
    avg_pool_bwd_op.set_attr<dims>("kernel", {2, 2});
    avg_pool_bwd_op.set_attr<dims>("pads_begin", {1, 1});
    avg_pool_bwd_op.set_attr<dims>("pads_end", {1, 1});
    avg_pool_bwd_op.set_attr<bool>("exclude_pad", true);

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t diff_src_lt = utils::logical_tensor_init(
            0, {1, 1, 4, 4}, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t diff_dst_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, diff_dst_lt};
    std::vector<impl::logical_tensor_t> outputs {diff_src_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto avg_pool_bwd_kernel = op_factory.create_kernel(avg_pool_bwd_op);

    avg_pool_bwd_kernel->compile(&avg_pool_bwd_op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t diff_dst_ts(diff_dst_lt, diff_dst.data());
    impl::tensor_t diff_src_ts(outputs[0], diff_src.data());

    impl::stream_t &strm = get_stream();
    avg_pool_bwd_kernel->execute(
            &avg_pool_bwd_op, &strm, {src_ts, diff_dst_ts}, {diff_src_ts});
    strm.wait();
    for (size_t i = 0; i < diff_src.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_src[i], ref_diff_src[i]);
    }
}

TEST(operator_kernel, avg_pool_backward_include_pad) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            3.0, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> diff_src(src.size(), 0.0);
    test::vector<float> ref_diff_src {-0.25, 0.75, 0.75, 2.5, 1.0, 4.0, 4.0,
            2.0, 1.0, 4.0, 4.0, 2.0, 3.0, -1.25, -1.25, -3.0 / 4};
    test::vector<float> diff_dst {
            -1.0, 3.0, 10.0, 4.0, 16.0, 8.0, 12.0, -5.0, -3.0};

    impl::op_t avg_pool_bwd_op(impl::op_kind::AvgPoolBackprop);
    avg_pool_bwd_op.set_attr<dims>("strides", {2, 2});
    avg_pool_bwd_op.set_attr<dims>("kernel", {2, 2});
    avg_pool_bwd_op.set_attr<dims>("pads_begin", {1, 1});
    avg_pool_bwd_op.set_attr<dims>("pads_end", {1, 1});
    avg_pool_bwd_op.set_attr<bool>("exclude_pad", false);

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t diff_src_lt = utils::logical_tensor_init(
            0, {1, 1, 4, 4}, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t diff_dst_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, diff_dst_lt};
    std::vector<impl::logical_tensor_t> outputs {diff_src_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto avg_pool_bwd_kernel = op_factory.create_kernel(avg_pool_bwd_op);

    avg_pool_bwd_kernel->compile(&avg_pool_bwd_op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t diff_dst_ts(diff_dst_lt, diff_dst.data());
    impl::tensor_t diff_src_ts(outputs[0], diff_src.data());

    impl::stream_t &strm = get_stream();
    avg_pool_bwd_kernel->execute(
            &avg_pool_bwd_op, &strm, {src_ts, diff_dst_ts}, {diff_src_ts});
    strm.wait();
    for (size_t i = 0; i < diff_src.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_src[i], ref_diff_src[i]);
    }
}

TEST(operator_kernel, max_pool_backward_with_incides) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            3.0, -1.0, 0.0, 1.0, -2.0, -1.0, 4.0};
    test::vector<float> diff_src(src.size(), 0.0);
    test::vector<float> ref_diff_src {0.0, 0.0, 16.0, 0.0, 4.0, 0.0, 0.0, 0.0,
            0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0};

    test::vector<float> diff_dst {4.0, 16.0, 8.0, 12.0};
#if DNNL_GRAPH_WITH_SYCL
    test::vector<int32_t> indices {2, 0, 1, 3};
#else
    test::vector<uint8_t> indices {2, 0, 1, 3};
#endif

    impl::op_t max_pool_bwd_op(impl::op_kind::MaxPoolBackprop);

    max_pool_bwd_op.set_attr<dims>("strides", dims {2, 2});
    max_pool_bwd_op.set_attr<dims>("kernel", dims {2, 2});
    max_pool_bwd_op.set_attr<dims>("pads_begin", dims {0, 0});
    max_pool_bwd_op.set_attr<dims>("pads_end", dims {0, 0});
    max_pool_bwd_op.set_attr<dims>("dilations", {1, 1});

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t diff_src_lt = utils::logical_tensor_init(
            1, {1, 1, 4, 4}, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t diff_dst_lt
            = utils::logical_tensor_init(3, {1, 1, 2, 2}, impl::data_type::f32);
#if DNNL_GRAPH_WITH_SYCL
    impl::logical_tensor_t indices_lt
            = utils::logical_tensor_init(4, {1, 1, 2, 2}, impl::data_type::s32);
#else
    impl::logical_tensor_t indices_lt
            = utils::logical_tensor_init(4, {1, 1, 2, 2}, impl::data_type::u8);
#endif

    auto &op_factory = get_dnnl_kernel_registry();
    auto max_pool_bwd_kernel = op_factory.create_kernel(max_pool_bwd_op);
    std::vector<impl::logical_tensor_t> outputs {diff_src_lt};
    max_pool_bwd_kernel->compile(
            &max_pool_bwd_op, &eng, {src_lt, diff_dst_lt}, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t diff_dst_ts(diff_dst_lt, diff_dst.data());
    impl::tensor_t diff_src_ts(outputs[0], diff_src.data());
    impl::tensor_t indices_ts(indices_lt, indices.data());

    impl::stream_t &strm = get_stream();
    max_pool_bwd_kernel->execute(&max_pool_bwd_op, &strm,
            {src_ts, indices_ts, diff_dst_ts}, {diff_src_ts});
    strm.wait();

    for (size_t i = 0; i < diff_src.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_src[i], ref_diff_src[i]);
    }
}

TEST(operator_kernel, max_pool_backward_without_incides) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            3.0, -1.0, 0.0, 1.0, -2.0, -1.0, 4.0};
    test::vector<float> diff_src(src.size(), 0.0);
    test::vector<float> ref_diff_src {0.0, 0.0, 16.0, 0.0, 4.0, 0.0, 0.0, 0.0,
            0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0};

    test::vector<float> diff_dst {4.0, 16.0, 8.0, 12.0};

    impl::op_t max_pool_bwd_op(impl::op_kind::MaxPoolBackprop);

    max_pool_bwd_op.set_attr<dims>("strides", dims {2, 2});
    max_pool_bwd_op.set_attr<dims>("kernel", dims {2, 2});
    max_pool_bwd_op.set_attr<dims>("pads_begin", dims {0, 0});
    max_pool_bwd_op.set_attr<dims>("pads_end", dims {0, 0});
    max_pool_bwd_op.set_attr<dims>("dilations", {1, 1});

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t diff_src_lt = utils::logical_tensor_init(
            1, {1, 1, 4, 4}, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t diff_dst_lt
            = utils::logical_tensor_init(3, {1, 1, 2, 2}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> outputs {diff_src_lt};
    auto &op_factory = get_dnnl_kernel_registry();
    auto max_pool_bwd_kernel = op_factory.create_kernel(max_pool_bwd_op);
    max_pool_bwd_kernel->compile(
            &max_pool_bwd_op, &eng, {src_lt, diff_dst_lt}, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t diff_dst_ts(diff_dst_lt, diff_dst.data());
    impl::tensor_t diff_src_ts(outputs[0], diff_src.data());

    impl::stream_t &strm = get_stream();
    max_pool_bwd_kernel->execute(
            &max_pool_bwd_op, &strm, {src_ts, diff_dst_ts}, {diff_src_ts});
    strm.wait();

    for (size_t i = 0; i < diff_src.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_src[i], ref_diff_src[i]);
    }
}

TEST(operator_kernel, layernorm_training) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {2.0, 4.0, 5.0, 2.0, 3.0, 5.0};
    test::vector<float> scale {1.0, 2.0};
    test::vector<float> shift {0.0, 1.0};
    test::vector<float> ref_dst {-1.0, 3.0, 1.0, -1.0, -1.0, 3.0};
    test::vector<float> ref_mean {3.0, 3.5, 4.0};
    test::vector<float> ref_var {1.0, 2.25, 1.0};
    test::vector<float> dst(src.size(), 0.0);
    test::vector<float> mean(ref_mean.size(), 0.0);
    test::vector<float> var(ref_var.size(), 0.0);

    impl::op_t op(impl::op_kind::LayerNorm);

    op.set_attr<float>("epsilon", 0);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 3, 2}, impl::data_type::f32);
    impl::logical_tensor_t scale_lt
            = utils::logical_tensor_init(1, {2}, impl::data_type::f32);
    impl::logical_tensor_t shift_lt
            = utils::logical_tensor_init(2, {2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            3, {1, 3, 2}, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t mean_lt = utils::logical_tensor_init(
            4, {3}, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t variance_lt = utils::logical_tensor_init(
            5, {3}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src_lt, scale_lt, shift_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt, mean_lt, variance_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto kernel = op_factory.create_kernel(op);

    kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);
    ASSERT_EQ(outputs[1].layout_type, impl::layout_type::opaque);
    ASSERT_EQ(outputs[2].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t scale_ts(scale_lt, scale.data());
    impl::tensor_t shift_ts(shift_lt, shift.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());
    impl::tensor_t mean_ts(outputs[1], mean.data());
    impl::tensor_t var_ts(outputs[2], var.data());

    impl::stream_t &strm = get_stream();
    kernel->execute(&op, &strm, {src_ts, scale_ts, shift_ts},
            {dst_ts, mean_ts, var_ts});
    strm.wait();

    for (size_t i = 0; i < ref_dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }

    for (size_t i = 0; i < ref_mean.size(); ++i) {
        ASSERT_FLOAT_EQ(mean[i], ref_mean[i]);
        ASSERT_FLOAT_EQ(var[i], ref_var[i]);
    }
}

TEST(operator_kernel, layernorm_inference) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {2.0, 4.0, 3.0, 5.5, 5.0, 4.0, 1.0, 2.5};
    test::vector<float> scale {1.0, 2.0};
    test::vector<float> shift {0.0, 1.0};
    test::vector<float> ref_dst {-1.0, 3.0, -1.0, 3.0, 1.0, -1.0, -1.0, 3.0};
    test::vector<float> dst(src.size(), 0.0);

    impl::op_t op(impl::op_kind::LayerNorm);

    op.set_attr<float>("epsilon", 0);
    op.set_attr<bool>("keep_stats", false); //inference

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {2, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t scale_lt
            = utils::logical_tensor_init(1, {2}, impl::data_type::f32);
    impl::logical_tensor_t shift_lt
            = utils::logical_tensor_init(2, {2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            3, {2, 2, 2}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src_lt, scale_lt, shift_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto kernel = op_factory.create_kernel(op);

    kernel->compile(&op, &eng, inputs, outputs);

    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t scale_ts(scale_lt, scale.data());
    impl::tensor_t shift_ts(shift_lt, shift.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());

    impl::stream_t &strm = get_stream();
    kernel->execute(&op, &strm, {src_ts, scale_ts, shift_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref_dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, layernorm_inference_without_scale_shift) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {2.0, 4.0, 3.0, 5.5, 5.0, 4.0, 1.0, 2.5};
    test::vector<float> ref_dst {-1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0};
    test::vector<float> dst(src.size(), 0.0);

    impl::op_t op(impl::op_kind::LayerNorm);

    op.set_attr<float>("epsilon", 0);
    op.set_attr<bool>("keep_stats", false); //inference
    op.set_attr<bool>("use_affine", false);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {2, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            3, {2, 2, 2}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto kernel = op_factory.create_kernel(op);

    kernel->compile(&op, &eng, inputs, outputs);

    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());

    impl::stream_t &strm = get_stream();
    kernel->execute(&op, &strm, {src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref_dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, convert_data) {
    impl::engine_t &engine = get_engine();

    test::vector<int8_t> src {1, 2, 3, 4, 5, 6};
    test::vector<int32_t> ref_dst {1, 2, 3, 4, 5, 6};
    test::vector<int32_t> dst(src.size(), 0);

    impl::op_t convert_op(impl::op_kind::convert);

    // prepare input/output logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 3}, impl::data_type::s8);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(1, {1, 2, 3}, impl::data_type::s32);

    auto &op_factory = get_dnnl_kernel_registry();
    auto kernel = op_factory.create_kernel(convert_op);

    kernel->compile(&convert_op, &engine, {src_lt}, {dst_lt});

    impl::stream_t &stream = get_stream();
    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    kernel->execute(&convert_op, &stream, {src_ts}, {dst_ts});
    stream.wait();
    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, quantize_per_tensor) {
    impl::engine_t &engine = get_engine();
    if (engine.kind() == impl::engine_kind::gpu) return;

    impl::op_t quantize(impl::op_kind::Quantize);
    quantize.set_attr<std::vector<float>>("scales", {0.1f});
    quantize.set_attr<std::vector<int64_t>>("zps", {10});
    quantize.set_attr<std::string>("qtype", "per_tensor");
    quantize.set_attr<int64_t>("axis", 0);

    test::vector<float> src {-1.0, 0.0, 1.0, 2.0};
    test::vector<uint8_t> dst(src.size(), 0);

    // int8 = f32 / scales + zero_points
    test::vector<uint8_t> ref_dst {0, 10, 20, 30};

    // prepare input/output logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(1, {1, 2, 2}, impl::data_type::u8);

    auto &op_factory = get_dnnl_kernel_registry();
    auto op = op_factory.create_kernel(quantize);

    op->compile(&quantize, &engine, {src_lt}, {dst_lt});

    impl::stream_t &stream = get_stream();
    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    op->execute(&quantize, &stream, {src_ts}, {dst_ts});
    stream.wait();
    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, quantize_per_tensor_any) {
    impl::engine_t &engine = get_engine();
    if (engine.kind() == impl::engine_kind::gpu) return;

    impl::op_t quantize(impl::op_kind::Quantize);
    quantize.set_attr<std::vector<float>>("scales", {0.1f});
    quantize.set_attr<std::vector<int64_t>>("zps", {10});
    quantize.set_attr<std::string>("qtype", "per_tensor");
    quantize.set_attr<int64_t>("axis", 0);

    test::vector<float> src {-1.0, 0.0, 1.0, 2.0};
    test::vector<uint8_t> dst(src.size(), 0);

    // int8 = f32 / scales + zero_points
    test::vector<uint8_t> ref_dst {0, 10, 20, 30};

    // prepare input/output logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 2, 2}, impl::data_type::u8, impl::layout_type::any);

    auto &op_factory = get_dnnl_kernel_registry();
    auto op = op_factory.create_kernel(quantize);

    op->compile(&quantize, &engine, {src_lt}, {dst_lt});

    impl::stream_t &stream = get_stream();
    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    op->execute(&quantize, &stream, {src_ts}, {dst_ts});
    stream.wait();
    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, quantize_per_channel_symmetric) {
    impl::engine_t &engine = get_engine();
    if (engine.kind() == impl::engine_kind::gpu) return;

    impl::op_t quantize(impl::op_kind::Quantize);
    quantize.set_attr<std::vector<float>>("scales", {0.1f, 0.2f});
    quantize.set_attr<std::vector<int64_t>>("zps", {0, 0});
    quantize.set_attr<std::string>("qtype", "per_channel");
    quantize.set_attr<int64_t>("axis", 1);

    test::vector<float> src {-1.0, 0.0, 1.0, 2.0};
    test::vector<int8_t> dst(src.size(), 0);

    // int8 = f32 / scales
    test::vector<int8_t> ref_dst {-10, 0, 5, 10};

    // prepare input/output logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(1, {1, 2, 2}, impl::data_type::s8);

    auto &op_factory = get_dnnl_kernel_registry();
    auto op = op_factory.create_kernel(quantize);

    op->compile(&quantize, &engine, {src_lt}, {dst_lt});

    impl::stream_t &stream = get_stream();
    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    op->execute(&quantize, &stream, {src_ts}, {dst_ts});
    stream.wait();
    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, dequantize_per_tensor) {
    impl::engine_t &engine = get_engine();
    if (engine.kind() == impl::engine_kind::gpu) return;

    impl::op_t dequantize(impl::op_kind::Dequantize);
    dequantize.set_attr<std::vector<float>>("scales", {0.1f});
    dequantize.set_attr<std::vector<int64_t>>("zps", {10});
    dequantize.set_attr<std::string>("qtype", "per_tensor");
    dequantize.set_attr<int64_t>("axis", 0);

    test::vector<uint8_t> src {0, 10, 20, 30};
    test::vector<float> dst(src.size(), 0);

    // f32 = scales * (int8 - zero_points)
    test::vector<float> ref_dst {-1.0, 0.0, 1.0, 2.0};

    // prepare input/output logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 2}, impl::data_type::u8);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(1, {1, 2, 2}, impl::data_type::f32);

    auto &op_factory = get_dnnl_kernel_registry();
    auto op = op_factory.create_kernel(dequantize);

    op->compile(&dequantize, &engine, {src_lt}, {dst_lt});

    impl::stream_t &stream = get_stream();
    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    op->execute(&dequantize, &stream, {src_ts}, {dst_ts});
    stream.wait();
    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, dequantize_per_tensor_any) {
    impl::engine_t &engine = get_engine();
    if (engine.kind() == impl::engine_kind::gpu) return;

    impl::op_t dequantize(impl::op_kind::Dequantize);
    dequantize.set_attr<std::vector<float>>("scales", {0.1f});
    dequantize.set_attr<std::vector<int64_t>>("zps", {10});
    dequantize.set_attr<std::string>("qtype", "per_tensor");
    dequantize.set_attr<int64_t>("axis", 0);

    test::vector<uint8_t> src {0, 10, 20, 30};
    test::vector<float> dst(src.size(), 0);

    // f32 = scales * (int8 - zero_points)
    test::vector<float> ref_dst {-1.0, 0.0, 1.0, 2.0};

    // prepare input/output logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 2}, impl::data_type::u8);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 2, 2}, impl::data_type::f32, impl::layout_type::any);

    auto &op_factory = get_dnnl_kernel_registry();
    auto op = op_factory.create_kernel(dequantize);

    op->compile(&dequantize, &engine, {src_lt}, {dst_lt});

    impl::stream_t &stream = get_stream();
    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    op->execute(&dequantize, &stream, {src_ts}, {dst_ts});
    stream.wait();
    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, dequantize_per_channel_symmetric) {
    impl::engine_t &engine = get_engine();
    if (engine.kind() == impl::engine_kind::gpu) return;

    impl::op_t dequantize(impl::op_kind::Dequantize);
    dequantize.set_attr<std::vector<float>>("scales", {0.1f, 0.2f});
    dequantize.set_attr<std::vector<int64_t>>("zps", {0, 0});
    dequantize.set_attr<std::string>("qtype", "per_channel");
    dequantize.set_attr<int64_t>("axis", 1);

    test::vector<int8_t> src {0, 10, 20, 30};
    test::vector<float> dst(src.size(), 0);

    // f32 = scales * int8
    test::vector<float> ref_dst {0.0, 1, 4, 6};

    // prepare input/output logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 2}, impl::data_type::s8);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(1, {1, 2, 2}, impl::data_type::f32);

    auto &op_factory = get_dnnl_kernel_registry();
    auto op = op_factory.create_kernel(dequantize);

    op->compile(&dequantize, &engine, {src_lt}, {dst_lt});

    impl::stream_t &stream = get_stream();
    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    op->execute(&dequantize, &stream, {src_ts}, {dst_ts});
    stream.wait();
    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_EQ(dst[i], ref_dst[i]);
    }
}
