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

#ifndef BACKEND_DNNL_OPERATORS_MATMUL_HPP
#define BACKEND_DNNL_OPERATORS_MATMUL_HPP

#include <functional>
#include <set>
#include <utility>
#include <vector>
#include <unordered_map>

#include "backend/dnnl/tensor.hpp"
#include "bn_fusion.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

namespace matmul_fwd {
enum matmul_inputs { kSrc, kWeight, kBias };
enum fused_bn_inputs { kScale, kShift, kMean, kVariance };
enum matmul_outputs { kDst };
} // namespace matmul_fwd

struct matmul_op_set {
    /**
     * Check if matmul operator has bias add.
     *
     * @param kind operator kind
     * @return whether the operator has bias add
     */
    static bool with_bias(op_kind_t kind) {
        static const std::set<op_kind_t> with_bias_set {op_kind::matmul_bias,
                op_kind::matmul_bias_add, op_kind::matmul_bias_add_relu,
                op_kind::matmul_bias_bn, op_kind::matmul_bias_elu,
                op_kind::matmul_bias_hardtanh, op_kind::matmul_bias_relu6,
                op_kind::matmul_bias_relu, op_kind::matmul_bias_gelu,
                op_kind::matmul_bias_sigmoid};
        return with_bias_set.count(kind);
    }

    /**
     * Check if matmul operator fuses batchnorm
     *
     * @param kind operator kind
     * @return whether the operator fuses batchnorm
     */
    static bool fuse_batchnorm(op_kind_t kind) {
        return kind == op_kind::matmul_bias_bn;
    }

    /**
     * Check if matmul operator fuses add
     *
     * @param kind operator kind
     * @return whether the operator fused add
     */

    static bool fuse_add(op_kind_t kind) {
        static const std::set<op_kind_t> with_add_set {op_kind::matmul_bias_add,
                op_kind::matmul_bias_add_relu, op_kind::matmul_add,
                op_kind::matmul_add_gelu, op_kind::matmul_add_relu,
                op_kind::matmul_add_sigmoid};
        return with_add_set.count(kind);
    }

    /**
     * Check if matmul operator fuses activation relu
     *
     * @param kind operator kind
     * @return whether the operator fused activation relu
     */
    static bool fuse_eltwise(op_kind_t kind) {
        static const std::set<op_kind_t> with_eltwise_set {op_kind::matmul_relu,
                op_kind::matmul_elu, op_kind::matmul_hardtanh,
                op_kind::matmul_gelu, op_kind::matmul_sigmoid,
                op_kind::matmul_bias_elu, op_kind::matmul_bias_hardtanh,
                op_kind::matmul_bias_relu6, op_kind::matmul_bias_relu,
                op_kind::matmul_bias_gelu, op_kind::matmul_bias_sigmoid,
                op_kind::matmul_bias_add_relu, op_kind::matmul_add_gelu,
                op_kind::matmul_add_relu, op_kind::matmul_add_sigmoid};
        return with_eltwise_set.count(kind);
    }
};

struct matmul_forward : public dnnl::matmul, public kernel_base {
    using super = dnnl::matmul;

private:
    // cached pd is in this struct
    primitive_desc pd_;
    dnnl::matmul prim_;
    // cache expected data to avoid creating memory in every iteration
    tensor expected_src_;
    tensor expected_weights_;
    tensor expected_bias_;
    tensor expected_dst_;
    attr_t attr_;

    tensor updated_weights_;
    tensor updated_bias_;

    bool transpose_a_ {false};
    bool transpose_b_ {false};

    bool with_bias_ {false};
    bool with_add_ {false};
    bool with_post_sum_ {false};
    bool with_post_binary_add_ {false};
    bool with_eltwise_ {false};
    bool with_bn_ {false};
    op_kind_t kind_;
    float alpha_ = 0.f;
    float beta_ = 0.f;

    size_t bn_input_offset_;

    float epsilon_; // bn epsilon

    tensor::desc ori_src_desc_ {};
    tensor::desc ori_weight_desc_ {};
    tensor::desc ori_bias_desc_ {};

    dnnl::engine p_engine_;
    dnnl::stream p_stream_;

public:
    impl::status_t compile_impl(const impl::op_t *op,
            const impl::engine_t *g_engine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        using desc = tensor::desc;
        kind_ = op->get_kind();
        with_bias_ = matmul_op_set::with_bias(kind_);
        with_add_ = matmul_op_set::fuse_add(kind_);
        with_eltwise_ = matmul_op_set::fuse_eltwise(kind_);
        with_bn_ = matmul_op_set::fuse_batchnorm(kind_);

        // deal with 1D add
        bool add_1d = (with_bias_ == false) && (with_add_ == true)
                && (impl::logical_tensor_wrapper(inputs[matmul_fwd::kBias])
                                .ndims()
                        == 1);
        if (add_1d) {
            with_bias_ = true;
            with_add_ = false;
        }

        // set attrs of eltwise
        if (with_eltwise_) {
            if (op->has_attr("alpha")) {
                alpha_ = op->get_attr<float>("alpha");
            } else if (op->has_attr("min")) {
                alpha_ = op->get_attr<float>("min");
            }

            if (op->has_attr("beta")) {
                beta_ = op->get_attr<float>("beta");
            } else if (op->has_attr("max")) {
                beta_ = op->get_attr<float>("max");
            }
        }
        // the bn inputs offset (if exist)
        if (with_bn_) bn_input_offset_ = with_bias_ ? 3 : 2;

        // prepare the inputs and outputs tensors' descs
        desc src {inputs.at(matmul_fwd::kSrc)};
        desc weight {inputs.at(matmul_fwd::kWeight)};

        dims old_weights_dims = weight.get_dims();

        //change dims and strides if tensor need to transpose
        if (op->has_attr("transpose_a"))
            transpose_a_ = op->get_attr<bool>("transpose_a");
        if (op->has_attr("transpose_b"))
            transpose_b_ = op->get_attr<bool>("transpose_b");

        if (transpose_a_ && src.get_ndims() > 1) {
            int ndims = src.get_ndims();
            dims expected_strides = src.get_strides();
            dims expected_dims = src.get_dims();
            const auto last_dim = static_cast<dims::size_type>(ndims - 1);
            std::swap(expected_dims[last_dim - 1], expected_dims[last_dim]);
            std::swap(
                    expected_strides[last_dim - 1], expected_strides[last_dim]);
            src = desc {expected_dims, src.get_data_type(), expected_strides};
        }

        if (transpose_b_ && weight.get_ndims() > 1) {
            int ndims = weight.get_ndims();
            dims expected_strides = weight.get_strides();
            dims expected_dims = weight.get_dims();
            const auto last_dim = static_cast<dims::size_type>(ndims - 1);
            std::swap(expected_dims[last_dim - 1], expected_dims[last_dim]);
            std::swap(
                    expected_strides[last_dim - 1], expected_strides[last_dim]);
            weight = desc {
                    expected_dims, weight.get_data_type(), expected_strides};
        }

        //if src or weight is 1-D, reshape it into 2-D for oneDNN
        if (src.get_ndims() == 1)
            src = desc {{1, src.get_dim(0)}, src.get_data_type(),
                    {src.get_dim(0), 1}};
        if (weight.get_ndims() == 1)
            weight = desc {
                    {weight.get_dim(0), 1}, weight.get_data_type(), {1, 1}};

        // if with_bias, we use the bias_desc directly, otherwise
        // if with_bn, we use bn's shift_desc as the bias_desc
        desc bias = with_bias_ ? desc {inputs.at(matmul_fwd::kBias)}
                               : (with_bn_ ? desc {inputs.at(bn_input_offset_
                                          + matmul_fwd::kShift)}
                                           : desc {});

        dims old_bias_dims = with_bias_ ? bias.get_dims() : dims {};

        impl::logical_tensor_t *dst_lt = const_cast<impl::logical_tensor_t *>(
                &outputs.at(matmul_fwd::kDst));

        tensor::desc dst(*dst_lt);

        // check the input dimension
        int src_ndims = src.get_ndims();
        int weight_ndims = weight.get_ndims();
        int bias_ndims = bias.get_ndims();
        int dst_ndims = dst.get_ndims();

        // expand src or weight for broadcast
        if (src_ndims != weight_ndims) {
            if (src_ndims > weight_ndims) {
                weight = expand(weight, src_ndims);
            } else {
                src = expand(src, weight_ndims);
            }
        }

        // if bias has different dims with dst, expand
        if (with_bias_ && bias_ndims != dst_ndims) {
            bias = expand(bias, dst_ndims);
        }

        ori_src_desc_ = src;
        ori_weight_desc_ = weight;
        ori_bias_desc_ = bias;

        if (with_bn_) epsilon_ = op->get_attr<float>("epsilon");

        // append post_ops to attrs
        if (with_add_) {
            impl::logical_tensor_t post_src_lt = inputs.back();
            if (impl::logical_tensor_wrapper(post_src_lt)
                            .has_same_shape_as(dst_lt)) {
                // if post src has the same shape of dst
                // set post sum attribute
                attr_ = attr_t::fuse_sum();
                if (with_eltwise_) {
                    attr_ = attr_t::residual(
                            get_eltwise_algo(kind_), 1.f, 1.f, alpha_, beta_);
                }
                with_post_sum_ = true;
            } else {
                desc post_src {post_src_lt};
                post_src = expand(post_src, dst.get_ndims());

                // post binary only supports per tensor and per channel
                // broadcast, which means the expand shape of post src should
                // be all one or the post_src_dim[1]==dst_dim[1]
                for (int i = dst.get_ndims() - 1; i >= 0; i--) {
                    if (post_src.get_dim(i) == 1) continue;

                    if (i != 1 || dst.get_dim(i) != post_src.get_dim(i)) {
                        return impl::status::compile_fail;
                    }
                }

                attr_ = attr_t::fuse_binary(post_src, algorithm::binary_add);
                with_post_binary_add_ = true;
            }
        } else if (with_eltwise_) {
            attr_ = attr_t::fuse_eltwise(
                    get_eltwise_algo(kind_), 1.f, alpha_, beta_);
        }

        p_engine_ = make_dnnl_engine(*g_engine);

        if (with_bias_) {
            BACKEND_DNNL_ENFORCE(
                    utils::one_of(bias.get_data_type(), data_type::f32,
                            data_type::f16, data_type::bf16),
                    "Incorrect data type in bias");
            bias = bias.to_format_any();
        }

        dims old_dst_dims = dst.get_dims();

        pd_ = with_bias_
                ? primitive_desc({src, weight, bias, dst}, attr_, p_engine_)
                : primitive_desc({src, weight, dst}, attr_, p_engine_);
        prim_ = super(pd_);

        fill_layout_info(dst_lt, pd_.dst_desc());

        // TODO(wuxun): for prepacking, temporarily skip when `with_bn_` is True
        // need to think about how to implement bn folding outside, maybe then
        // we can also remove `with_bn_` flag.
        if (!with_bn_) {
            impl::logical_tensor_t *ori_weight_lt
                    = const_cast<impl::logical_tensor_t *>(
                            &inputs.at(matmul_fwd::kWeight));
            // TODO(wuxun): here, we need to reshape the queried desc to the
            // original shape. However, if there is a broadcast in one dim and
            // DNNL also needs padding in this broadcast dim, reshaping will
            // fail. A possible solution is that in conversion's reorder, we
            // also add check for the broadcast-able dims.
            fill_layout_info(ori_weight_lt, pd_.weights_desc());
            if (with_bias_) {
                impl::logical_tensor_t *ori_bias_lt
                        = const_cast<impl::logical_tensor_t *>(
                                &inputs.at(matmul_fwd::kBias));
                fill_layout_info(
                        ori_bias_lt, pd_.bias_desc().reshape(old_bias_dims));
            }
        }
        return impl::status::success;
    }

    impl::status_t execute_impl(const impl::op_t *op,
            const impl::stream_t *g_stream,
            const std::vector<impl::tensor_t> &inputs,
            const std::vector<impl::tensor_t> &outputs) override {
        UNUSED(op);
        p_stream_ = make_dnnl_stream(p_engine_, *g_stream);
        impl::allocator_t *alc = g_stream->get_engine()->get_allocator();

        auto pd_src_desc = pd_.src_desc();
        auto pd_weights_desc = pd_.weights_desc();
        auto pd_bias_desc = pd_.bias_desc();
        auto pd_dst_desc = pd_.dst_desc();

        // always parse the inputs buffer with processed desc
        tensor src {ori_src_desc_, p_engine_, alc,
                inputs.at(matmul_fwd::kSrc).get_data_handle()};
        tensor weight {ori_weight_desc_, p_engine_, alc,
                inputs.at(matmul_fwd::kWeight).get_data_handle()};

        tensor bias = with_bias_ ? tensor {ori_bias_desc_, p_engine_, alc,
                              inputs.at(matmul_fwd::kBias).get_data_handle()}
                                 : tensor {};

        tensor post_src = with_add_ ? tensor {inputs.back(), p_engine_, alc}
                                    : tensor {};
        tensor dst {outputs.at(matmul_fwd::kDst), p_engine_, alc};
        if (with_bn_) {
            const tensor bn_scale {
                    inputs.at(bn_input_offset_ + matmul_fwd::kScale), p_engine_,
                    alc};
            const tensor bn_shift {
                    inputs.at(bn_input_offset_ + matmul_fwd::kShift), p_engine_,
                    alc};
            const tensor bn_mean {
                    inputs.at(bn_input_offset_ + matmul_fwd::kMean), p_engine_,
                    alc};
            const tensor bn_var {
                    inputs.at(bn_input_offset_ + matmul_fwd::kVariance),
                    p_engine_, alc};

            if (updated_weights_.is_empty()) {
                updated_weights_ = tensor {weight.get_desc(), p_engine_, alc};
                updated_bias_ = tensor {bn_shift.get_desc(), p_engine_, alc};
            }

            bn_fusion::folding(&updated_weights_, &updated_bias_, weight, bias,
                    bn_mean, bn_var, bn_scale, bn_shift, epsilon_, *g_stream);

            if (updated_weights_.get_desc()
                    != pd_weights_desc) { //need to reorder
                if (expected_weights_.is_empty()) {
                    expected_weights_
                            = tensor {pd_weights_desc, p_engine_, alc};
                }
                updated_weights_.reorder_to(p_stream_, expected_weights_);
            } else {
                expected_weights_ = updated_weights_;
            }

            if (updated_bias_.get_desc() != pd_bias_desc) {
                if (expected_bias_.is_empty()) {
                    expected_bias_ = tensor {pd_bias_desc, p_engine_, alc};
                }
                updated_bias_.reorder_to(p_stream_, expected_bias_);
            } else {
                expected_bias_ = updated_bias_;
            }

        } else {
            if (weight.get_desc() != pd_weights_desc) { //need to reorder
                if (expected_weights_.is_empty()) {
                    expected_weights_
                            = tensor {pd_weights_desc, p_engine_, alc};
                }
                weight.reorder_to(p_stream_, expected_weights_);
            } else {
                expected_weights_ = weight;
            }

            if (with_bias_) {
                if (bias.get_desc() != pd_bias_desc) {
                    if (expected_bias_.is_empty()) {
                        expected_bias_ = tensor {pd_bias_desc, p_engine_, alc};
                    }
                    bias.reorder_to(p_stream_, expected_bias_);
                } else {
                    expected_bias_ = bias;
                }
            }
        }

        if (src.get_desc() != pd_src_desc) {
            if (expected_src_.is_empty()) {
                expected_src_ = tensor {pd_src_desc, p_engine_, alc};
            }
            src.reorder_to(p_stream_, expected_src_);
        } else {
            expected_src_ = src;
        }

        if (dst.get_desc() != pd_dst_desc) {
            if (expected_dst_.is_empty()) {
                expected_dst_ = tensor {pd_dst_desc, p_engine_, alc};
            }
        } else {
            expected_dst_ = dst;
        }

        if (with_post_sum_
                && post_src.get_data_handle()
                        != expected_dst_.get_data_handle()) {
            post_src.reorder_to(p_stream_, expected_dst_);
        }

        std::unordered_map<int, memory> matmul_args;
        matmul_args.insert({DNNL_ARG_SRC, expected_src_});
        matmul_args.insert({DNNL_ARG_WEIGHTS, expected_weights_});
        matmul_args.insert({DNNL_ARG_DST, expected_dst_});

        if (with_bias_) { matmul_args.insert({DNNL_ARG_BIAS, expected_bias_}); }

        if (with_post_binary_add_) {
            matmul_args.insert(
                    {(DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1),
                            post_src});
        }

        prim_.execute(p_stream_, matmul_args);

        if (expected_dst_ != dst) expected_dst_.reorder_to(p_stream_, dst);
        return impl::status::success;
    }

private:
    algorithm get_eltwise_algo(op_kind_t kind) {
        switch (static_cast<int>(kind)) {
            case op_kind::matmul_relu:
            case op_kind::matmul_bias_relu:
            case op_kind::matmul_bias_add_relu:
            case op_kind::matmul_add_relu: return (algorithm::eltwise_relu);

            case op_kind::matmul_elu:
            case op_kind::matmul_bias_elu: return (algorithm::eltwise_elu);

            case op_kind::matmul_hardtanh:
            case op_kind::matmul_bias_relu6:
            case op_kind::matmul_bias_hardtanh:
                return (algorithm::eltwise_clip);

            case op_kind::matmul_sigmoid:
            case op_kind::matmul_bias_sigmoid:
            case op_kind::matmul_add_sigmoid:
                return (algorithm::eltwise_logistic);

            case op_kind::matmul_gelu:
            case op_kind::matmul_bias_gelu:
            case op_kind::matmul_add_gelu: return (algorithm::eltwise_gelu_erf);

            default:
                BACKEND_DNNL_ENFORCE(
                        0, "Unsupported fused_eltwise op for matmul.");
        }
        return algorithm::undef;
    }

    impl::status_t prepare_inplace_pairs_impl(const impl::engine_t *g_engine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        UNUSED(g_engine);
        if (with_post_sum_) {
            size_t input_idx = with_bias_ ? matmul_fwd::kBias + 1
                                          : matmul_fwd::kWeight + 1;
            if (with_bn_)
                input_idx = bn_input_offset_ + matmul_fwd::kVariance + 1;
            constexpr size_t output_idx = 0;

            const logical_tensor_wrapper post_src_lt(inputs[input_idx]);
            const logical_tensor_wrapper dst_lt(outputs[output_idx]);
            if (post_src_lt.is_opaque() && dst_lt.is_opaque()
                    && post_src_lt.is_similar(dst_lt))
                inplace_pairs_.push_back(
                        {inputs[input_idx].id, outputs[output_idx].id});
        }
        return impl::status::success;
    }
};

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
