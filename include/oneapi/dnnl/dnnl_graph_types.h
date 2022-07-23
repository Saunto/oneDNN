/*******************************************************************************
 * Copyright 2020-2022 Intel Corporation
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

/// @file
/// C API definitions

#ifndef ONEAPI_DNNL_DNNL_GRAPH_TYPES_H
#define ONEAPI_DNNL_DNNL_GRAPH_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

/// @cond DO_NOT_DOCUMENT_THIS
#include <stddef.h>
#include <stdint.h>

#include "oneapi/dnnl/dnnl_types.h"
/// @endcond

/// @addtogroup dnnl_graph_api
/// @{

/// @addtogroup dnnl_graph_api_logical_tensor
/// @{

/// A wildcard value for number of dimensions which is unknown at a tensor or
/// operation creation time.
#define DNNL_GRAPH_UNKNOWN_NDIMS -1

/// A wildcard value for dimensions that are unknown at a tensor or operation
/// creation time.
#define DNNL_GRAPH_UNKNOWN_DIM -1

/// Layout type specification
typedef enum {
    /// Undefined layout type
    dnnl_graph_layout_type_undef = 0,
    /// Any means to let the library to decide the layout for a tensor during
    /// partition compilation.
    dnnl_graph_layout_type_any = 1,
    /// Strided means that the layout of a tensor is determined by the strides
    /// field in the logical tensor.
    dnnl_graph_layout_type_strided = 2,
    /// Opaque means that the layout of a tensor is the library specific.
    /// Usually, an opaque layout is generated by a partition which is compiled
    /// with layout type any.
    dnnl_graph_layout_type_opaque = 3,
} dnnl_graph_layout_type_t;

/// Logical tensor property
typedef enum {
    /// Undefined tensor property
    dnnl_graph_tensor_property_undef = 0,
    /// Variable means the tensor may be changed during computation or between
    /// different iterations.
    dnnl_graph_tensor_property_variable = 1,
    /// Constant means the tensor will keep unchanged during computation and
    /// between different iterations. It's useful for the library to apply
    /// optimizations for constant tensors or cache constant tensors inside the
    /// library. For example, constant weight tensors in inference scenarios.
    dnnl_graph_tensor_property_constant = 2,
} dnnl_graph_tensor_property_t;

/// Logical tensor. It is based on an ID, a number of dimensions, dimensions
/// themselves, element data type, tensor property and tensor memory layout.
typedef struct {
    /// Unique id of each logical tensor. The library uses logical tensor IDs to
    /// build up the connections between operations if the output of one
    /// operation has the same ID as the input of another operation.
    size_t id;

    /// Number of dimensions. -1 means unknown (DNNL_GRAPH_UNKNOWN_NDIMS). 0 is
    /// used to define scalar tensor.
    int32_t ndims;

    /// Size of each dimension. -1 means the size of that dimension is unknown.
    /// 0 is used to define zero-dimension tensor. The library supports to
    /// deduce output shapes according to input shapes during compilation.
    /// Unlike memory descriptor in oneDNN primitive API, the order of
    /// dimensions is not defined in logical tensor. It is defined by the
    /// operations which respect the order through the attributes
    /// #dnnl_graph_op_attr_data_format or #dnnl_graph_op_attr_filter_format.
    /// For example, for a Convolution with `data_format=NXC`, it means the
    /// first element of dims of activation tensor is mini-batch size, the last
    /// effective element of dims is channel size, and other elements between
    /// them are spatial dimensions.
    dnnl_dims_t dims;

    /// Data type of the tensor elements.
    dnnl_data_type_t data_type;

    /// Property type of the tensor.
    dnnl_graph_tensor_property_t property;

    /// Layout type of the tensor.
    dnnl_graph_layout_type_t layout_type;
    union {
        /// The field is valid when `layout_type` is
        /// #dnnl_graph_layout_type_strided. -1 means the stride of the
        /// dimension is unknown. The library currently doesn't support other
        /// negative stride values.
        dnnl_dims_t strides;

        /// The field is valid when `layout_type` is
        /// #dnnl_graph_layout_type_opaque. An opaque layout ID is usually
        /// generated by a partition which is compiled with layout type any.
        size_t layout_id;
    } layout;
} dnnl_graph_logical_tensor_t;

/// @} dnnl_graph_api_logical_tensor

/// @addtogroup dnnl_graph_api_partition
/// @{

/// Policy specifications for partitioning
typedef enum {
    /// Max policy is to be defined. The library intends to deliver best
    /// optimization and larger partition with max policy. It also means users
    /// may lose fine-grained control the operations in the partition.
    /// Currently, max policy has the same effect as fusion policy.
    dnnl_graph_partition_policy_max = 0,
    /// Fusion policy returns partitions with typical post-op fusions, eg.
    /// Convolution + ReLU or other element-wise operations or a chian of
    /// post-ops.
    dnnl_graph_partition_policy_fusion = 1,
    /// Debug policy doesn't not apply any fusions. It returns partitions with
    /// single operation in each partition. The policy is useful when users
    /// notice any bug or correctness issue in max policy or fusion policy.
    dnnl_graph_partition_policy_debug = 2,
} dnnl_graph_partition_policy_t;

/// Partition kind. It defines the basic structure of the subgraph contained in
/// a partition. For example, kind
/// #dnnl_graph_partition_kind_convolution_post_ops indicates the partition
/// contains one Convolution and its post-ops. But the operation kind of the
/// post-ops are not specified. Partition's kind is decided by the library
/// internally and can be queried from a partition.
typedef enum {
    /// The partition kind is not defined.
    dnnl_graph_partition_kind_undef = 0,
    /// The partition contains a Convolution and its post-ops.
    dnnl_graph_partition_kind_convolution_post_ops = 1,
    /// The partition contains a ConvTranspose and its post-ops.
    dnnl_graph_partition_kind_convtranspose_post_ops = 2,
    /// The partition contains an Interpolate and its post-ops.
    dnnl_graph_partition_kind_interpolate_post_ops = 3,
    /// The partition contains a MatMul and its post-ops.
    dnnl_graph_partition_kind_matmul_post_ops = 4,
    /// The partition contains a Reduction and its post-ops.
    dnnl_graph_partition_kind_reduction_post_ops = 5,
    /// The partition contains an Unary op and its post-ops.
    dnnl_graph_partition_kind_unary_post_ops = 6,
    /// The partition contains a Binary op and its post-ops.
    dnnl_graph_partition_kind_binary_post_ops = 7,
    /// The partition contains a Pooling op (AvgPool or MaxPool) and its
    /// post-ops.
    dnnl_graph_partition_kind_pooling_post_ops = 8,
    /// The partition contains a BatchNorm op and its post-ops.
    dnnl_graph_partition_kind_batch_norm_post_ops = 9,
    /// Other partitions based on post-ops but not specified by above kinds.
    dnnl_graph_partition_kind_misc_post_ops = 10,
    /// The partition contains a quantized version of Convolution and its
    /// post-ops.
    dnnl_graph_partition_kind_quantized_convolution_post_ops = 11,
    /// The partition contains a quantized version of ConvTranspose and its
    /// post-ops.
    dnnl_graph_partition_kind_quantized_convtranspose_post_ops = 12,
    /// The partition contains a quantized version of MatMul and its
    /// post-ops.
    dnnl_graph_partition_kind_quantized_matmul_post_ops = 13,
    /// The partition contains a quantized version of Unary op and its
    /// post-ops.
    dnnl_graph_partition_kind_quantized_unary_post_ops = 14,
    /// The partition contains a quantized version of Pooling op and its
    /// post-ops.
    dnnl_graph_partition_kind_quantized_pooling_post_ops = 15,
    /// Other partitions based quantization and post-ops but not specified
    /// by above kinds.
    dnnl_graph_partition_kind_misc_quantized_post_ops = 16,
    /// The partition contains a Convolution backward op and its post-ops.
    dnnl_graph_partition_kind_convolution_backprop_post_ops = 17,
    /// The partition contains a variant of Multi-head Attention.
    dnnl_graph_partition_kind_mha = 18,
    /// The partition contains a variant of Multi-layer Perceptron.
    dnnl_graph_partition_kind_mlp = 19,
    /// The partition contains a variant of quantized MHA.
    dnnl_graph_partition_kind_quantized_mha = 20,
    /// The partition contains a variant of quantized MLP.
    dnnl_graph_partition_kind_quantized_mlp = 21,
    /// The partition contains a variant of residual Convolution block with
    /// multiple Convolutions in it.
    dnnl_graph_partition_kind_residual_conv_blocks = 22,
    /// The partition contains a variant of quantized version of residual
    /// Convolution block with multiple Convolutions in it.
    dnnl_graph_partition_kind_quantized_residual_conv_blocks = 23,
} dnnl_graph_partition_kind_t;

/// An opaque structure to describe a partition.
struct dnnl_graph_partition;

/// A partition handle.
typedef struct dnnl_graph_partition *dnnl_graph_partition_t;

/// A constant partition handle.
typedef const struct dnnl_graph_partition *const_dnnl_graph_partition_t;

/// @} dnnl_graph_api_partition

/// @addtogroup dnnl_graph_api_graph
/// @{

/// An opaque structure to describe a graph.
struct dnnl_graph_graph;

/// A graph handle.
typedef struct dnnl_graph_graph *dnnl_graph_graph_t;

/// A constant graph handle.
typedef const struct dnnl_graph_graph *const_dnnl_graph_graph_t;

/// @}

/// @addtogroup dnnl_graph_api_op
/// @{

/// Kinds of operations
typedef enum {
    dnnl_graph_op_abs,
    dnnl_graph_op_abs_backprop,
    dnnl_graph_op_add,
    dnnl_graph_op_avg_pool,
    dnnl_graph_op_avg_pool_backprop,
    dnnl_graph_op_batch_norm_backprop,
    dnnl_graph_op_batch_norm_forward_training,
    dnnl_graph_op_batch_norm_inference,
    dnnl_graph_op_bias_add,
    dnnl_graph_op_bias_add_backprop,
    dnnl_graph_op_clamp,
    dnnl_graph_op_clamp_backprop,
    dnnl_graph_op_concat,
    dnnl_graph_op_convolution,
    dnnl_graph_op_convolution_backprop_data,
    dnnl_graph_op_convolution_backprop_filters,
    dnnl_graph_op_conv_transpose,
    dnnl_graph_op_conv_transpose_backprop_data,
    dnnl_graph_op_conv_transpose_backprop_filters,
    dnnl_graph_op_dequantize,
    dnnl_graph_op_divide,
    dnnl_graph_op_dynamic_dequantize,
    dnnl_graph_op_dynamic_quantize,
    dnnl_graph_op_dynamic_reshape,
    dnnl_graph_op_dynamic_transpose,
    dnnl_graph_op_elu,
    dnnl_graph_op_elu_backprop,
    dnnl_graph_op_end,
    dnnl_graph_op_equal,
    dnnl_graph_op_erf,
    dnnl_graph_op_exp,
    dnnl_graph_op_gelu,
    dnnl_graph_op_gelu_backprop,
    dnnl_graph_op_greater,
    dnnl_graph_op_greater_equal,
    dnnl_graph_op_hard_swish,
    dnnl_graph_op_hard_swish_backprop,
    dnnl_graph_op_index,
    dnnl_graph_op_interpolate,
    dnnl_graph_op_interpolate_backprop,
    dnnl_graph_op_layer_norm,
    dnnl_graph_op_layer_norm_backprop,
    dnnl_graph_op_leaky_relu,
    dnnl_graph_op_less,
    dnnl_graph_op_less_equal,
    dnnl_graph_op_log,
    dnnl_graph_op_log_softmax,
    dnnl_graph_op_log_softmax_backprop,
    dnnl_graph_op_logical_and,
    dnnl_graph_op_logical_not,
    dnnl_graph_op_logical_or,
    dnnl_graph_op_logical_xor,
    dnnl_graph_op_matmul,
    dnnl_graph_op_maximum,
    dnnl_graph_op_max_pool,
    dnnl_graph_op_max_pool_backprop,
    dnnl_graph_op_minimum,
    dnnl_graph_op_mish,
    dnnl_graph_op_mish_backprop,
    dnnl_graph_op_multiply,
    dnnl_graph_op_negative,
    dnnl_graph_op_not_equal,
    dnnl_graph_op_pow,
    dnnl_graph_op_pow_backprop,
    dnnl_graph_op_pow_backprop_exponent,
    dnnl_graph_op_prelu,
    dnnl_graph_op_prelu_backprop,
    dnnl_graph_op_quantize,
    dnnl_graph_op_reciprocal,
    dnnl_graph_op_reduce_l1,
    dnnl_graph_op_reduce_l2,
    dnnl_graph_op_reduce_max,
    dnnl_graph_op_reduce_mean,
    dnnl_graph_op_reduce_min,
    dnnl_graph_op_reduce_prod,
    dnnl_graph_op_reduce_sum,
    dnnl_graph_op_relu,
    dnnl_graph_op_relu_backprop,
    dnnl_graph_op_reorder,
    dnnl_graph_op_round,
    dnnl_graph_op_select,
    dnnl_graph_op_sigmoid,
    dnnl_graph_op_sigmoid_backprop,
    dnnl_graph_op_sign,
    dnnl_graph_op_softmax,
    dnnl_graph_op_softmax_backprop,
    dnnl_graph_op_softplus,
    dnnl_graph_op_softplus_backprop,
    dnnl_graph_op_sqrt,
    dnnl_graph_op_sqrt_backprop,
    dnnl_graph_op_square,
    dnnl_graph_op_squared_difference,
    dnnl_graph_op_static_reshape,
    dnnl_graph_op_static_transpose,
    dnnl_graph_op_subtract,
    dnnl_graph_op_tanh,
    dnnl_graph_op_tanh_backprop,
    dnnl_graph_op_type_cast,
    dnnl_graph_op_wildcard,
    dnnl_graph_op_last_symbol,
} dnnl_graph_op_kind_t;

/// Attributes of operations
typedef enum {
    /// Undefined op attribute.
    dnnl_graph_op_attr_undef = 0,

    // float32 attributes. The value of these attributes can be any single
    // float32 number.

    /// Specifies an alpha attribute to an op.
    dnnl_graph_op_attr_alpha = 1,
    /// Specifies an beta attribute to an op.
    dnnl_graph_op_attr_beta,
    /// Specifies an epsilon attribute to an op.
    dnnl_graph_op_attr_epsilon,
    /// Specifies a max attribute to an op.
    dnnl_graph_op_attr_max,
    ///Specifies a min attribute to an op.
    dnnl_graph_op_attr_min,
    /// Specifies a momentum attribute to an op.
    dnnl_graph_op_attr_momentum,

    // float32 vector attributes. The value of these attributes can be a vector
    // of float32 numbers.

    /// Specifies a scales attribute to an op.
    dnnl_graph_op_attr_scales,

    // int64_t attributes. The value of these attributes can be any single int64
    // number.

    /// Specifies an axis attribute to an op.
    dnnl_graph_op_attr_axis = 0x20,
    /// Specifies a begin_norm_axis attribute to an op.
    dnnl_graph_op_attr_begin_norm_axis,
    /// Specifies a groups attribute to an op.
    dnnl_graph_op_attr_groups,

    // int64_t vector attributes. The value of these attributes can be a vector
    // of int64 numbers.

    /// Specifies an axes attribute to an op.
    dnnl_graph_op_attr_axes,
    /// Specifies a dilations attribute to an op.
    dnnl_graph_op_attr_dilations,
    /// Specifies a filter_shape attribute to an op.
    dnnl_graph_op_attr_filter_shape,
    /// Specifies a input_shape attribute to an op.
    dnnl_graph_op_attr_input_shape,
    /// Specifies a kernel attribute to an op.
    dnnl_graph_op_attr_kernel,
    /// Specifies an order attribute to an op.
    dnnl_graph_op_attr_order,
    /// Specifies an output_padding attribute to an op.
    dnnl_graph_op_attr_output_padding,
    /// Specifies an output_shape attribute to an op.
    dnnl_graph_op_attr_output_shape,
    /// Specifies a pads_begin attribute to an op.
    dnnl_graph_op_attr_pads_begin,
    /// Specifies a pads_end attribute to an op.
    dnnl_graph_op_attr_pads_end,
    /// Specifies a shape attribute to an op.
    dnnl_graph_op_attr_shape,
    /// Specifies a sizes attribute to an op.
    dnnl_graph_op_attr_sizes,
    /// Specifies a strides attribute to an op.
    dnnl_graph_op_attr_strides,
    /// Specifies a zps attribute to an op.
    dnnl_graph_op_attr_zps,

    // bool attributes. The value of these attributes can be any single bool
    // value.

    /// Specifies an exclude_pad attribute to an op.
    dnnl_graph_op_attr_exclude_pad = 0x40,
    /// Specifies a keep_dims attribute to an op.
    dnnl_graph_op_attr_keep_dims,
    /// Specifies a keep_stats attribute to an op.
    dnnl_graph_op_attr_keep_stats,
    /// Specifies a per_channel_broadcast attribute to an op.
    dnnl_graph_op_attr_per_channel_broadcast,
    /// Specifies a special_zero attribute to an op.
    dnnl_graph_op_attr_special_zero,
    /// Specifies a transpose_a attribute to an op.
    dnnl_graph_op_attr_transpose_a,
    /// Specifies a transpose_b attribute to an op.
    dnnl_graph_op_attr_transpose_b,
    /// Specifies an use_affine attribute to an op.
    dnnl_graph_op_attr_use_affine,
    /// Specifies an use_dst attribute to an op.
    dnnl_graph_op_attr_use_dst,

    // string attributes. The value of these attributes can be a string.

    /// Specifies an auto_broadcast attribute to an op. The value can be "none"
    /// or "numpy".
    dnnl_graph_op_attr_auto_broadcast = 0x60,
    /// Specifies an auto_pad attribute to an op. The value can be "none",
    /// "same_upper", "same_lower", or "valid".
    dnnl_graph_op_attr_auto_pad,
    /// Specifies an coordinate_transformation_mode attribute to an op. The
    /// value can be "half_pixel" or "align_corners". The attribute is defined
    /// for Interpolate operations.
    dnnl_graph_op_attr_coordinate_transformation_mode,
    /// Specifies a data_format of an op. The value can be "NCX" or "NXC".
    dnnl_graph_op_attr_data_format,
    /// Specifies a filter_format of an op. The value can be "OIX" or "XIO".
    dnnl_graph_op_attr_filter_format,
    /// Specifies a mode attribute of an op. The value can be "nearest",
    /// "linear", "bilinear", or "trilinear". The attribute is defined for
    /// Interpolate operations.
    dnnl_graph_op_attr_mode,
    /// Specifies a qtype attribute to an op. The value can be "per_channel" or
    /// "per_tensor". The attribute is defined for quantization operations.
    dnnl_graph_op_attr_qtype,
    /// Specifies a rounding_type attribute to an op. The value can be "ceil" or
    /// "floor".
    dnnl_graph_op_attr_rounding_type,
} dnnl_graph_op_attr_t;

/// An opaque structure to describe an operation.
struct dnnl_graph_op;

/// An operation handle.
typedef struct dnnl_graph_op *dnnl_graph_op_t;

/// A constant operation handle.
typedef const struct dnnl_graph_op *const_dnnl_graph_op_t;

/// @} dnnl_graph_api_op

/// @addtogroup dnnl_graph_api_allocator
/// @{

/// Allocation call-back function interface for CPU
typedef void *(*dnnl_graph_host_allocate_f)(size_t size, size_t alignment);

/// Deallocation call-back function interface for CPU
typedef void (*dnnl_graph_host_deallocate_f)(void *);

/// Allocation call-back function interface for SYCL device
typedef void *(*dnnl_graph_sycl_allocate_f)(
        size_t size, size_t alignment, const void *dev, const void *context);

/// brief Deallocation call-back function interface for SYCL device
typedef void (*dnnl_graph_sycl_deallocate_f)(
        void *buf, const void *dev, const void *context, void *event);

/// An opaque structure to describe an allocator.
struct dnnl_graph_allocator;

/// An allocator handle.
typedef struct dnnl_graph_allocator *dnnl_graph_allocator_t;

/// A constant allocator handle.
typedef const struct dnnl_graph_allocator *const_dnnl_graph_allocator_t;

/// @} dnnl_graph_api_allocator

/// @addtogroup dnnl_graph_api_compiled_partition
/// @{

/// In-place pair definition. It can queried from a compiled partition
/// indicating that an input and an output of the partition can share the same
/// memory buffer for computation. In-place computation helps to reduce the
/// memory footprint and improves cache locality. But since the library may not
/// have a global view of user's application, it's possible that the tensor with
/// `input_id` is used at other places in user's computation graph. In this
/// case, the user should take the in-place pair as a hint and pass a different
/// memory buffer for output tensor to avoid overwriting the input memory buffer
/// which will probably cause unexpected incorrect results.
typedef struct {
    /// The id of input tensor
    size_t input_id;

    /// The id of output tensor
    size_t output_id;
} dnnl_graph_inplace_pair_t;

/// An opaque structure to describe a compiled partition.
struct dnnl_graph_compiled_partition;

/// A compiled partition handle.
typedef struct dnnl_graph_compiled_partition *dnnl_graph_compiled_partition_t;

/// A constant compiled partition handle.
typedef const struct dnnl_graph_compiled_partition
        *const_dnnl_graph_compiled_partition_t;

/// @} dnnl_graph_api_compiled_partition

/// @addtogroup dnnl_graph_api_tensor
/// @{

/// An opaque structure to describe a tensor.
struct dnnl_graph_tensor;

/// A tensor handle.
typedef struct dnnl_graph_tensor *dnnl_graph_tensor_t;

/// A constant tensor handle.
typedef const struct dnnl_graph_tensor *const_dnnl_graph_tensor_t;

/// @} dnnl_graph_api_tensor

/// @} dnnl_graph_api

#ifdef __cplusplus
}
#endif
#endif
