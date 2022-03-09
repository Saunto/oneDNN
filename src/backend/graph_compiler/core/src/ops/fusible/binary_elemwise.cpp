/*******************************************************************************
 * Copyright 2022 Intel Corporation
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

#include <assert.h>

#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include "binary_elemwise.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/graph/fusible_op.hpp>
#include <compiler/ir/graph/fusible_op_utils.hpp>
#include <compiler/ir/graph/fusion_mgr.hpp>
#include <util/utils.hpp>

namespace sc {

std::vector<int> binary_elementwise_op_impl_t::infer_broadcast_axis() const {
    int bc_input_idx = get_broadcast_input();
    if (bc_input_idx == -1) return {};

    sc_dims lhs_dims, rhs_dims;
    lhs_dims = get_inputs()[0]->details_.get_plain_dims();
    rhs_dims = get_inputs()[1]->details_.get_plain_dims();

    sc_dims elt_dims, bc_dims;
    if (bc_input_idx == 1) {
        elt_dims = lhs_dims;
        bc_dims = rhs_dims;
    } else {
        elt_dims = rhs_dims;
        bc_dims = lhs_dims;
    }
    if (bc_dims.size() == 1 && bc_dims[0] == 1) {
        return std::vector<int> {-1};
    }
    std::vector<int> bc_axis;
    // broad-cast conditions 1: the shape of lhs and rhs not match
    if (elt_dims.size() != bc_dims.size()) {
        std::vector<int> common_axes(elt_dims.size(), 0);
        // from right to left
        int64_t i = elt_dims.size();
        for (int64_t j = bc_dims.size() - 1; j >= 0; j--) {
            while (i >= 1) {
                i--;
                if (elt_dims.at(i) == bc_dims.at(j)) {
                    common_axes.at(i) = 1;
                    break;
                }
            }
            if (i == -1) {
                COMPILE_ASSERT(0,
                        "illegal elementwise operand found. "
                                << utils::print_vector(elt_dims) << " , "
                                << utils::print_vector(bc_dims));
            }
        }
        for (size_t j = 0; j < common_axes.size(); ++j)
            if (common_axes.at(j) == 1) bc_axis.emplace_back(j);
    }
    // broad-cast conditions 2: the shape of lhs and rhs match,
    // but length=1 in dims
    else {
        bool double_check_broadcast = false;
        for (size_t i = 0; i < elt_dims.size(); ++i) {
            if (elt_dims.at(i) != bc_dims.at(i)) {
                if (bc_dims.at(i) == 1) {
                    double_check_broadcast = true;
                } else {
                    COMPILE_ASSERT(0,
                            "illegal elementwise operand found: "
                                    << utils::print_vector(elt_dims) << " , "
                                    << utils::print_vector(bc_dims));
                }
            }
        }
        if (double_check_broadcast) {
            for (size_t i = 0; i < elt_dims.size(); ++i) {
                if (elt_dims.at(i) == bc_dims.at(i)) {
                    bc_axis.emplace_back(i);
                }
            }
            if (bc_axis.empty()) { bc_axis.emplace_back(-1); }
        } else
            bc_axis = {};
    }
    return bc_axis;
}

void infer_binary_slice_ranges(
        fusible_op_t *cur, fslice_map &fsmap, infer_status_map_t &stat_map) {
    COMPILE_ASSERT(cur->get_inputs().size() == 2, "binary op is expected");
    // search known ranges from any input of cur fusbile op
    slice_range_map known_ranges_map = search_known_slice_ranges(cur, fsmap);
    auto &outslice = fsmap.get(cur->get_outputs()[0]);
    // if unkown slice ranges exist.
    if (known_ranges_map.size() < cur->get_inputs().size()) {
        int unknown_idx
                = known_ranges_map.find(0) != known_ranges_map.end() ? 1 : 0;
        known_ranges_map[unknown_idx] = known_ranges_map[1 - unknown_idx];
        // set the other unknown slice range by achieved known_ranges_list
        set_unknown_slice_ranges(cur, known_ranges_map, fsmap, stat_map);
    }
    // set outputs slice range
    outslice = known_ranges_map[0];
}

static slice_range_list infer_broadcast_arg_slice(
        slice_range_list known_range_list, std::vector<int> bc_axis,
        bool keep_dims) {
    slice_range_list bc_arg_range_list(known_range_list.size());
    for (size_t i = 0; i < bc_arg_range_list.size(); i++) {
        auto &known_range = known_range_list[i];
        for (size_t j = 0; j < known_range.size(); j++) {
            if (bc_axis.end() != std::find(bc_axis.begin(), bc_axis.end(), j)) {
                bc_arg_range_list[i].emplace_back(known_range.at(j));
            } else {
                if (keep_dims) {
                    bc_arg_range_list[i].emplace_back(
                            std::make_pair(expr(0), expr(1)));
                }
            }
        }
        if (bc_arg_range_list[i].empty())
            bc_arg_range_list[i].emplace_back(std::make_pair(0, 1));
    }
    return bc_arg_range_list;
}

static slice_range_list infer_broadcast_slice(slice_range_list known_range_list,
        std::vector<int> bc_axis, sc_dims bc_dim) {
    slice_range_list bc_range_list(known_range_list.size());
    for (size_t i = 0; i < bc_range_list.size(); i++) {
        auto &known_range = known_range_list[i];
        COMPILE_ASSERT(known_range.size() == bc_dim.size()
                        || bc_axis == std::vector<int> {-1},
                "Unexpected cases found")
        for (size_t j = 0; j < known_range.size(); j++) {
            if (bc_axis.end() != std::find(bc_axis.begin(), bc_axis.end(), j)) {
                bc_range_list[i].emplace_back(known_range.at(j));
            } else {
                bc_range_list[i].emplace_back(
                        std::make_pair(expr(0), dim2unsigned(bc_dim[j])));
            }
        }
    }
    return bc_range_list;
}

binary_elementwise_op_impl_t::binary_elementwise_op_impl_t(
        const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs) {
    // TODO(xxx): do not cache vectorized_ or inplace_
    assert(ins.size() == 2);
    info_.inputs_ = ins;
    if (outs.empty()) {
        // fixme: correctly infer the shape for broadcast
        auto lhs_const = dynamic_cast<constant_op_t *>(
                info_.inputs_.at(0)->producer_owner_);
        auto rhs_const = dynamic_cast<constant_op_t *>(
                info_.inputs_.at(1)->producer_owner_);
        if (!lhs_const && rhs_const) {
            info_.outputs_.emplace_back(
                    std::make_shared<graph_tensor>(this, ins[0]->details_));
        } else if (lhs_const && !rhs_const) {
            info_.outputs_.emplace_back(
                    std::make_shared<graph_tensor>(this, ins[1]->details_));
        } else {
            int bc_input_idx = get_broadcast_input();
            int ref_idx = bc_input_idx < 0 ? 0 : 1 - bc_input_idx;
            info_.outputs_.emplace_back(std::make_shared<graph_tensor>(
                    this, ins[ref_idx]->details_));
        }
    } else {
        info_.outputs_ = outs;
    }

    int bc_idx = get_broadcast_input();
    int non_bc_idx = bc_idx < 0 ? 0 : 1 - bc_idx;

    info_.outputs_[0]->details_.dtype_
            = info_.inputs_[non_bc_idx]->details_.dtype_;

    attrs_ = attrs;
    plain_bc_axis_ = attrs.get_or_else("bc_axis", std::vector<int> {});

    if (plain_bc_axis_.empty()) { plain_bc_axis_ = infer_broadcast_axis(); }

    inplace_ = attrs.get_or_else("inplace", 1);
}

binary_elementwise_op_impl_t::binary_elementwise_op_impl_t(graph_tensor_ptr lhs,
        graph_tensor_ptr rhs, elt_operator elt_op, int inplace)
    : binary_elementwise_op_impl_t(
            {std::move(lhs), std::move(rhs)}, {}, {{"inplace", inplace}}) {
    elt_op_ = elt_op;
    switch (elt_op) {
        case elt_operator::ADD: op_name_ = "add"; break;
        case elt_operator::SUB: op_name_ = "sub"; break;
        case elt_operator::MUL: op_name_ = "mul"; break;
        case elt_operator::DIV: op_name_ = "div"; break;
        case elt_operator::MIN: op_name_ = "min"; break;
        case elt_operator::MAX: op_name_ = "max"; break;
        case elt_operator::SQD_DIFF: op_name_ = "sqd_diff"; break;
        default: break;
    }
}

int binary_elementwise_op_impl_t::get_broadcast_input() const {
    const sc_dims &lhs_dims = info_.inputs_[0]->details_.get_plain_dims();
    const sc_dims &rhs_dims = info_.inputs_[1]->details_.get_plain_dims();
    if (lhs_dims == rhs_dims) {
        return -1;
    } else {
        auto lhs_dp = get_dims_product(lhs_dims);
        auto rhs_dp = get_dims_product(rhs_dims);
        if (lhs_dp == rhs_dp) {
            COMPILE_ASSERT(lhs_dims.size() != rhs_dims.size(),
                    "Unexpected dims of bianry elementwise inputs are found: "
                            << utils::print_vector(lhs_dims) << " and "
                            << utils::print_vector(rhs_dims))
            return lhs_dims.size() > rhs_dims.size() ? 1 : 0;
        } else {
            return lhs_dp > rhs_dp ? 1 : 0;
        }
    }
}

static sc_data_format_t infer_broadcast_format(
        const logical_tensor_t &target_lt, const logical_tensor_t &bc_lt) {
    COMPILE_ASSERT(
            bc_lt.get_plain_dims().size() == target_lt.get_plain_dims().size(),
            "infer_blocking_format only support plain dimension aligned cases");
    sc_data_format_kind_t target_lt_format_code
            = target_lt.get_format().format_code_;
    sc_data_format_t::blocking_t blocks = target_lt.get_format().blocks_;
    sc_data_format_kind_t bc_lt_format_code = bc_lt.get_format().format_code_;
    // start infer the blocks
    sc_dims bc_plain_dim = bc_lt.get_plain_dims();
    int block_dim = target_lt_format_code.ndims()
            - target_lt_format_code.norig_dims();
    int target_batch_dim = target_lt.get_plain_dims().size()
            - target_lt_format_code.norig_dims();
    for (int i = target_lt_format_code.norig_dims();
            i < target_lt_format_code.ndims(); ++i) {
        int blocking_axis = target_lt_format_code.get(i);
        // if the axis's corresponding plain dim is 1
        // blocks should also be 1
        if (bc_plain_dim[target_batch_dim + blocking_axis] == 1) {
            blocks[i - target_lt_format_code.norig_dims()] = 1;
        }
    }
    // start infer the format code
    if (target_lt_format_code.is_batch_format()
            == bc_lt_format_code.is_batch_format()) {
        // if both batch OR both non-batch
        // smaller side's format code == larger side's format code
        COMPILE_ASSERT(target_lt_format_code.norig_dims()
                        == bc_lt_format_code.norig_dims(),
                "Unsupported case for binary_elementwise query format.");
        return sc_data_format_t(target_lt.get_format().format_code_, blocks);
    } else {
        // if one side is batch and another is non-batch
        // needs to cast the dimension axis inside the storage args
        if (target_lt_format_code.is_batch_format()
                && !bc_lt_format_code.is_batch_format()) {
            int dim_difference = bc_lt_format_code.norig_dims()
                    - target_lt_format_code.norig_dims();
            assert(dim_difference >= 0);
            std::vector<int> bc_lt_storage_args(
                    target_lt_format_code.ndims() + dim_difference, -1);
            std::iota(bc_lt_storage_args.begin(), bc_lt_storage_args.end(), 0);
            for (int i = 0; i < target_lt_format_code.ndims(); ++i) {
                bc_lt_storage_args[i + dim_difference]
                        = target_lt_format_code.get(i) + dim_difference;
            }
            return sc_data_format_t(false, bc_lt_storage_args, blocks);
        } else {
            int dim_difference = bc_lt_format_code.norig_dims()
                    - target_lt_format_code.norig_dims();
            std::vector<int> bc_lt_storage_args(
                    bc_lt_format_code.norig_dims() + block_dim,
                    -1); // ensure reorder is convertiable
            for (int i = 0; i < target_lt_format_code.ndims(); ++i) {
                if (i + dim_difference >= 0) {
                    COMPILE_ASSERT(
                            target_lt_format_code.get(i) + dim_difference >= 0,
                            "Unsupported format encountered in "
                            "binary_elementwise query format.");
                    bc_lt_storage_args[i + dim_difference]
                            = target_lt_format_code.get(i) + dim_difference;
                }
            }
            if (std::find(bc_lt_storage_args.begin(), bc_lt_storage_args.end(),
                        -1)
                    != bc_lt_storage_args.end()) {
                COMPILE_ASSERT(0,
                        "Unsupported format encountered in "
                        "binary elementwise query format.");
            }
            return sc_data_format_t(true, bc_lt_storage_args, blocks);
        }
    }
    return sc_data_format_t(target_lt.get_format().format_code_);
}

void binary_elementwise_op_impl_t::query_format(context_ptr ctx,
        std::vector<std::vector<sc_data_format_t>> &in_formats,
        std::vector<std::vector<sc_data_format_t>> &out_formats) {
    auto in0_format = info_.inputs_[0]->details_.get_format();
    auto in1_format = info_.inputs_[1]->details_.get_format();

    int bc_input_idx = get_broadcast_input();

    if (info_.inputs_[0]->details_.get_plain_dims().size()
            != info_.inputs_[1]->details_.get_plain_dims().size()) {
        COMPILE_ASSERT(in0_format == sc_data_format_t(format_kinds::A)
                        || in1_format == sc_data_format_t(format_kinds::A),
                "Unsupported format encountered in binary elementwise query "
                "format.");
        in_formats.push_back({in0_format});
        in_formats.push_back({in1_format});
        out_formats.push_back({!bc_input_idx ? in1_format : in0_format});
    } else {
        if (!bc_input_idx) {
            auto target_format = infer_broadcast_format(
                    info_.inputs_[1]->details_, info_.inputs_[0]->details_);
            in_formats.push_back({target_format});
            in_formats.push_back({in1_format});
            out_formats.push_back({in1_format});
        } else {
            auto target_format = infer_broadcast_format(
                    info_.inputs_[0]->details_, info_.inputs_[1]->details_);
            in_formats.push_back({in0_format});
            in_formats.push_back({target_format});
            out_formats.push_back({in0_format});
        }
    }
}

void binary_elementwise_op_impl_t::prepare_fusion_data(fdata_map &fdmap) {
    COMPILE_ASSERT(!op_name_.empty(), "op_name or elt_operator is not set.\n");
    COMPILE_ASSERT(info_.outputs_.size() == 1, "Wrong op output size.\n");
    auto &output = info_.outputs_[0];
    auto &outdetail = fdmap.get(output);
    auto &in_detail0 = fdmap.get(info_.inputs_[0]);
    auto &in_detail1 = fdmap.get(info_.inputs_[1]);

    in_detail0.use_count_++;
    in_detail1.use_count_++;
    auto lhs_const = dynamic_cast<constant_op_t *>(
            info_.inputs_.at(0)->producer_owner_);
    auto rhs_const = dynamic_cast<constant_op_t *>(
            info_.inputs_.at(1)->producer_owner_);
    // no inplace, need to create a new buffer
    if (inplace_ == 0) {
        info_.tensor_share_info_ = {};
    }
    // inplace 1-th input
    else if (inplace_ == 1) {
        if (lhs_const
                || (info_.inputs_[0]->details_.get_blocking_dims()
                        != output->details_.get_blocking_dims())) {
            info_.tensor_share_info_ = {};
            inplace_ = 0;
        } else {
            info_.tensor_share_info_ = {{0, {0}}};
        }
    }
    // inplace 2-th input
    else if (inplace_ == 2) {
        if (rhs_const
                || (info_.inputs_[1]->details_.get_blocking_dims()
                        != output->details_.get_blocking_dims())) {
            info_.tensor_share_info_ = {};
            inplace_ = 0;
        } else {
            info_.tensor_share_info_ = {{0, {1}}};
        }
    } else {
        COMPILE_ASSERT(0,
                "binary op only have two inputs, but got "
                        << inplace_ << "-th input to be inplaced.");
    }
}

// The logic below might be suitable for most fusible op, which has same
// slice ranges on inputs and outputs
void binary_elementwise_op_impl_t::infer_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {
    COMPILE_ASSERT(get_inputs().size() == 2, "binary op is expected");
    // search known ranges from any input of cur fusbile op
    slice_range_map known_ranges_map = search_known_slice_ranges(this, fsmap);
    auto &outslice = fsmap.get(get_outputs()[0]);
    // if unkown slice ranges exist.
    if (known_ranges_map.size() < get_inputs().size()) {
        int unknown_idx
                = known_ranges_map.find(0) != known_ranges_map.end() ? 1 : 0;
        // check broadcast
        int bc_input_idx = get_broadcast_input();
        if (bc_input_idx >= 0) {
            bool keep_dims = get_inputs()[bc_input_idx]
                                     ->details_.get_blocking_dims()
                                     .size()
                    == get_inputs()[1 - bc_input_idx]
                               ->details_.get_blocking_dims()
                               .size();
            auto bc_axis = get_bc_axis();
            if (unknown_idx != bc_input_idx) {
                slice_range_list bc_range_list = infer_broadcast_slice(
                        known_ranges_map[1 - unknown_idx], bc_axis,
                        get_inputs()[1 - bc_input_idx]
                                ->details_.get_blocking_dims());
                known_ranges_map[unknown_idx] = bc_range_list;
            } else {
                slice_range_list bc_arg_range_list = infer_broadcast_arg_slice(
                        known_ranges_map[1 - unknown_idx], bc_axis, keep_dims);
                known_ranges_map[unknown_idx] = bc_arg_range_list;
            }
            // set the other unknown slice range by achieved
            // known_ranges_list
            set_unknown_slice_ranges(this, known_ranges_map, fsmap, stat_map);
            // set outputs slice range
            outslice = known_ranges_map[1 - bc_input_idx];
            return;
        } else {
            known_ranges_map[unknown_idx] = known_ranges_map[1 - unknown_idx];
        }
        // set the other unknown slice range by achieved known_ranges_list
        set_unknown_slice_ranges(this, known_ranges_map, fsmap, stat_map);
    }
    // set outputs slice range
    int bc_idx = get_broadcast_input();
    if (bc_idx != -1 && bc_idx == inplace_ - 1) {
        // inplace_ is 0 or 1 or 2
        // if we have inplace_, but the inplace_ side is the smaller side
        // we need to follow the bc_idx rather than inplace_ to set outslice
        outslice = known_ranges_map[1 - bc_idx];
    } else {
        outslice = known_ranges_map[inplace_ == 2 ? 1 : 0];
    }
}

void binary_elementwise_op_impl_t::pre_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {
    auto &outslice = fsmap.get(get_outputs()[0]);
    // check broadcast
    int bc_input_idx = get_broadcast_input();
    for (size_t i = 0; i < get_inputs().size(); i++) {
        auto &input = get_inputs()[i];
        auto &inpslice = fsmap.get(input);
        if (inpslice.empty()) {
            if (bc_input_idx == static_cast<int>(i)) {
                auto bc_axis = get_bc_axis();
                inpslice = infer_broadcast_arg_slice(outslice, bc_axis,
                        get_inputs()[bc_input_idx]
                                        ->details_.get_blocking_dims()
                                        .size()
                                == get_inputs()[1 - bc_input_idx]
                                           ->details_.get_blocking_dims()
                                           .size());
            } else {
                inpslice = outslice;
            }
            input->producer_owner_->dyn_cast<fusible_op_t>()->pre_slice_ranges(
                    fsmap, stat_map);
        }
    }
}

std::vector<int> binary_elementwise_op_impl_t::get_bc_axis() const {
    int bc_input_idx = get_broadcast_input();
    if (bc_input_idx == -1) return {};
    if (plain_bc_axis_ == std::vector<int> {-1}) return plain_bc_axis_;
    return transform_axis_plain2blocking(
            info_.inputs_[1 - bc_input_idx], plain_bc_axis_);
}

bool binary_elementwise_op_impl_t::register_brgemm_fusion(
        const context_ptr &ctx, const std::vector<tensor_slice *> &outputs,
        const std::vector<const tensor_slice *> &inputs,
        brgemm_fusion_register &brg_reg) {
    if (!fuse_in_brgemm_) { return false; }
    int bc_input_idx = get_broadcast_input();
    // input 0 broadcast, can not be processed in brgemm
    if (bc_input_idx == 0) { return false; }
    return brg_reg.register_op_infos(shared_from_this(),
            outputs[0]->get_tensor_ptr(), inputs[1]->get_tensor_ptr(),
            inputs[1]->get_shape());
}

sc_dims binary_elementwise_op_impl_t::get_bwise_fuse_shrink_dims() const {
    auto &in0_detail = info_.inputs_[0]->details_;
    auto &in1_detail = info_.inputs_[1]->details_;
    if (in0_detail.get_format().format_code_.is_batch_format()
            != in1_detail.get_format().format_code_.is_batch_format()) {
        return {};
    }
    auto output_dims = info_.outputs_[0]->details_.get_blocking_dims();
    int offset = op_traits::batchwise_shrinkable_t::get_shrinkable_offset(
            info_.outputs_[0]);
    return {output_dims.begin(), output_dims.begin() + offset};
}

void binary_elementwise_op_impl_t::collect_shrinked_lt_map(
        int bw_size, gt2gt_map &bw_lt_map) {
    int bc_idx = get_broadcast_input();
    if (bc_idx == -1)
        op_traits::batchwise_shrinkable_t::collect_shrinked_lt_map(
                bw_size, bw_lt_map);
    std::vector<graph_tensor_ptr> new_ins;
    op_traits::batchwise_shrinkable_t::record_shrinked_gt(
            bw_lt_map, get_outputs()[0], bw_size);
    auto old_ins = get_inputs();
    bool keep_dims = get_inputs()[0]->details_.get_blocking_dims().size()
            == get_inputs()[1]->details_.get_blocking_dims().size();
    auto bc_axis = get_bc_axis();
    int valid_size = 0;
    for (auto &ax : bc_axis) {
        if (ax < bw_size)
            valid_size++;
        else
            break;
    }
    for (size_t i = 0; i < get_inputs().size(); i++) {
        op_traits::batchwise_shrinkable_t::record_shrinked_gt(bw_lt_map,
                get_inputs()[i],
                static_cast<int>(i) == bc_idx && !keep_dims ? valid_size
                                                            : bw_size);
    }
}

void binary_elementwise_op_impl_t::collect_shrinked_axes_map(
        int bw_size, gt2axes_map &bw_axes_map) {
    int bc_idx = get_broadcast_input();
    if (bc_idx == -1)
        op_traits::batchwise_shrinkable_t::collect_shrinked_axes_map(
                bw_size, bw_axes_map);
    std::vector<graph_tensor_ptr> new_ins;
    op_traits::batchwise_shrinkable_t::record_shrinked_axes(
            bw_axes_map, get_outputs()[0], bw_size);
    auto old_ins = get_inputs();
    bool keep_dims = get_inputs()[0]->details_.get_blocking_dims().size()
            == get_inputs()[1]->details_.get_blocking_dims().size();
    auto bc_axis = get_bc_axis();
    std::vector<int> bw_axes;
    for (int i = 0; i < bw_size; i++) {
        auto iter = std::find(bc_axis.begin(), bc_axis.end(), i);
        if (iter != bc_axis.end()) {
            bw_axes.emplace_back(iter - bc_axis.begin());
        } else {
            bw_axes.emplace_back(-1);
        }
    }
    for (size_t i = 0; i < get_inputs().size(); i++) {
        if (static_cast<int>(i) == bc_idx && !keep_dims) {
            op_traits::batchwise_shrinkable_t::record_shrinked_axes(
                    bw_axes_map, get_inputs()[i], bw_axes);
        } else {
            op_traits::batchwise_shrinkable_t::record_shrinked_axes(
                    bw_axes_map, get_inputs()[i], bw_size);
        }
    }
}

void compute_block_broadcast(const std::vector<const tensor_slice *> &src,
        const tensor_slice &dst, int bc_input_idx,
        const std::vector<int> &bc_axis, const vectorized_info_t &vx_info,
        const std::function<expr(expr, expr)> &compute,
        sc_data_type_t dtype = datatypes::f32, size_t wkld = 0UL) {
    // nested loop vars
    std::vector<expr> iter_vars;
    // the indices for multiple inputs. First dim: the input, Second dim: the
    // dimemsions in the tensor
    std::vector<expr> in_idx, in_bc_idx;
    // the indices for the output tensor
    std::vector<expr> dst_idx;

    COMPILE_ASSERT(bc_input_idx == 0 || bc_input_idx == 1,
            "bc_input_idx is expected to be 0 or 1")
    const tensor_slice *in_tsl = src[1 - bc_input_idx],
                       *in_bc_tsl = src[bc_input_idx];
    bool keep_dims = in_tsl->get_base_dims().size()
            == in_bc_tsl->get_base_dims().size();
    // add output type check, manual downcast
    sc_data_etype out_etype
            = dst.tptr_->dtype_.get_pointer_element().as_etype();
    // use src_indices.at(0) as default
    for (unsigned i = 0; i < dst.nslice_dims(); i++) {
        // make the loop var for the for-loop
        iter_vars.emplace_back(builder::make_var(datatypes::index,
                std::string("_fuseiter") + fusion_create_idx()));
        in_idx.emplace_back(iter_vars.back());
        if (std::find(bc_axis.begin(), bc_axis.end(), i) != bc_axis.end()) {
            in_bc_idx.emplace_back(iter_vars.back());
        } else if (keep_dims) {
            in_bc_idx.emplace_back(0);
        }
        /** push an index for output tensor **/
        dst_idx.emplace_back(iter_vars.back());
    }
    // For empty bc_axis
    if (in_bc_idx.empty()) in_bc_idx = {0};
    std::vector<expr> in_idx_tail = in_idx, in_bc_idx_tail = in_bc_idx,
                      dst_idx_tail = dst_idx;
    auto tail_var = builder::make_var(
            datatypes::index, std::string("_fuseiter") + fusion_create_idx());
    in_idx_tail[vx_info.axis] = tail_var;
    dst_idx_tail[vx_info.axis] = tail_var;

    expr indexed_target
            = builder::make_indexing(dst.tptr_, dst_idx, vx_info.lanes);
    expr indexed_input
            = builder::make_indexing(in_tsl->tptr_, in_idx, vx_info.lanes);

    expr indexed_target_tail = builder::make_indexing(dst.tptr_, dst_idx_tail);
    expr indexed_input_tail
            = builder::make_indexing(in_tsl->tptr_, in_idx_tail);
    if (!in_tsl->tptr_->dtype_.get_pointer_element().is_etype(out_etype)) {
        indexed_input = builder::make_cast(
                sc_data_type_t(out_etype, indexed_input->dtype_.lanes_),
                indexed_input);
        indexed_input_tail = builder::make_cast(
                sc_data_type_t(out_etype, indexed_input_tail->dtype_.lanes_),
                indexed_input);
    }
    auto bld = builder::get_current_builder();
    COMPILE_ASSERT(bld, "No active builder is set");
    auto slice_len = get_const_as_int(
            dst.get_shape().at(vx_info.axis).static_as<constant>());
    int floor = slice_len / vx_info.lanes * vx_info.lanes;
    int tail = slice_len % vx_info.lanes;
    std::vector<stmt> tcur;
    stmt cur;
    bool bc_input_cast
            = !in_bc_tsl->tptr_->dtype_.get_pointer_element().is_etype(
                    out_etype);
    // recover schedule loop
    for (int i = static_cast<int>(dst.get_shape().size() - 1); i >= 0; i--) {
        stmt body;
        // move broadcast op to body
        if (static_cast<int>(dst.get_shape().size()) == vx_info.axis + 1
                && i == vx_info.axis) {
            // IF last dim is included in bc_axis.
            if (floor) {
                expr indexed_bc_input;
                if (bc_axis.back() == static_cast<int64_t>(vx_info.axis)) {
                    indexed_bc_input = builder::make_indexing(
                            in_bc_tsl->tptr_, in_bc_idx, vx_info.lanes);
                }
                // IF last dim is excluded in bc_axis.
                else {
                    indexed_bc_input = builder::make_broadcast(
                            builder::make_indexing(in_bc_tsl->tptr_, in_bc_idx),
                            static_cast<int>(vx_info.lanes));
                }
                if (bc_input_cast) {
                    indexed_bc_input = builder::make_cast(
                            sc_data_type_t(
                                    out_etype, indexed_bc_input->dtype_.lanes_),
                            indexed_bc_input);
                }
                bld->push_scope();
                cur = make_stmt<assign_node_t>(indexed_target,
                        bc_input_idx == 1
                                ? compute(indexed_input, indexed_bc_input)
                                : compute(indexed_bc_input, indexed_input));
                cur->attr()[op_traits::workload_computable_t::workload_number]
                        = wkld;
                bld->emit(cur);
                cur = make_stmt<for_loop_node_t>(iter_vars.at(i), expr(0),
                        expr(floor), expr(int(vx_info.lanes)), bld->pop_scope(),
                        true, for_type::NORMAL);
                tcur.emplace_back(cur);
            }
            if (tail) {
                auto res_it = std::find(
                        bc_axis.begin(), bc_axis.end(), vx_info.axis);
                if (res_it != bc_axis.end()) {
                    in_bc_idx_tail[keep_dims ? vx_info.axis
                                             : (res_it - bc_axis.begin())]
                            = tail_var;
                }
                expr indexed_bc_input_tail = builder::make_indexing(
                        in_bc_tsl->tptr_, in_bc_idx_tail);
                if (bc_input_cast) {
                    indexed_bc_input_tail = builder::make_cast(
                            sc_data_type_t(out_etype,
                                    indexed_bc_input_tail->dtype_.lanes_),
                            indexed_bc_input_tail);
                }
                bld->push_scope();
                cur = make_stmt<assign_node_t>(indexed_target_tail,
                        bc_input_idx == 1 ? compute(
                                indexed_input_tail, indexed_bc_input_tail)
                                          : compute(indexed_bc_input_tail,
                                                  indexed_input_tail));
                cur->attr()[op_traits::workload_computable_t::workload_number]
                        = wkld;
                bld->emit(cur);
                cur = make_stmt<for_loop_node_t>(tail_var, expr(floor),
                        expr(floor + tail), expr(1), bld->pop_scope(), true,
                        for_type::NORMAL);
                tcur.emplace_back(cur);
            }
        } else {
            if (!tcur.empty() && tcur[0].defined()) {
                body = make_stmt<stmts_node_t>(std::move(tcur));
                tcur.clear();
                // address special condition, like temp_buffer is used
                cur = make_stmt<for_loop_node_t>(std::move(iter_vars.at(i)),
                        expr(0), dst.get_shape().at(i), expr(1),
                        std::move(body), true, for_type::NORMAL);
            } else if (cur.defined()) {
                body = make_stmt<stmts_node_t>(
                        std::vector<stmt> {std::move(cur)});
                // address special condition, like temp_buffer is used
                cur = make_stmt<for_loop_node_t>(std::move(iter_vars.at(i)),
                        expr(0), dst.get_shape().at(i), expr(1),
                        std::move(body), true, for_type::NORMAL);
            } else {
                // if cur not defined, means last axis of tensor slice has range
                // 1, e.g. tensor_slice{{i, 100},{0, 1}}
                indexed_target = builder::make_indexing(dst.tptr_, dst_idx);

                indexed_input = builder::make_indexing(in_tsl->tptr_, in_idx);

                expr indexed_bc_input
                        = builder::make_indexing(in_bc_tsl->tptr_, in_bc_idx);
                if (bc_input_cast) {
                    indexed_bc_input = builder::make_cast(
                            sc_data_type_t(
                                    out_etype, indexed_bc_input->dtype_.lanes_),
                            indexed_bc_input);
                }
                bld->push_scope();
                cur = make_stmt<assign_node_t>(indexed_target,
                        bc_input_idx == 1
                                ? compute(indexed_input, indexed_bc_input)
                                : compute(indexed_bc_input, indexed_input));
                cur->attr()[op_traits::workload_computable_t::workload_number]
                        = wkld;
                bld->emit(cur);
                cur = make_stmt<for_loop_node_t>(iter_vars.at(i), expr(0),
                        dst.get_shape().at(i), expr(1), bld->pop_scope(), true,
                        for_type::NORMAL);
            }
        }
    }
    if (!tcur.empty() && tcur[0].defined()) {
        assert(dst.get_shape().size() == 1UL);
        // TODO(xxx): currenly we don't add merge_loop attribute for this
        // special case, need stronger loop analysis.
        for (auto &it : tcur) {
            bld->emit(it);
        }
    } else {
        cur->attr()[stmt_attr_key::merge_loop] = true;
        bld->emit(cur);
    }
}

void binary_elementwise_op_impl_t::compute_block(context_ptr ctx,
        const std::vector<tensor_slice *> &dst,
        const std::vector<const tensor_slice *> &inputs) {
    size_t wkld = compute_fusible_workload(ctx, dst, inputs);
    // set default vectorized information
    vx_info_.axis = dst[0]->get_shape().size() - 1;

    for (int64_t i = dst[0]->nslice_dims() - 1; i >= 0; --i) {
        int cur_dim = get_const_as_int(
                dst[0]->get_shape()[i].checked_as<constant>());
        if (1 != cur_dim) {
            vx_info_.axis = i;
            break;
        }
    }
    vx_info_.lanes
            = vectorize_step(ctx, info_.inputs_[0]->details_.dtype_.type_code_);

    // use broad-cast
    int bc_input_idx = get_broadcast_input();
    if (bc_input_idx != -1) {
        // reuse broadcast op
        compute_block_broadcast(
                inputs, *dst[0], bc_input_idx, get_bc_axis(), vx_info_,
                [&](const expr &in_0, const expr &in_1) -> expr {
                    switch (elt_op_) {
                        case elt_operator::ADD: return (in_0 + in_1);
                        case elt_operator::SUB: return (in_0 - in_1);
                        case elt_operator::MUL: return (in_0 * in_1);
                        case elt_operator::DIV: return (in_0 / in_1);
                        case elt_operator::MIN:
                            return builder::make_min(in_0, in_1);
                        case elt_operator::MAX:
                            return builder::make_max(in_0, in_1);
                        case elt_operator::SQD_DIFF:
                            return (in_0 - in_1) * (in_0 - in_1);
                        default:
                            COMPILE_ASSERT(
                                    false, "Unsupport elementwise op found.\n");
                            return expr();
                    }
                },
                info_.outputs_[0]->details_.dtype_, wkld);
    } else {
        auto func = [&](const std::vector<expr> &in,
                            std::vector<expr::lvalue_proxy_t> &out,
                            int mask_count, float mask_value) -> stmt {
            auto out_dtype = out[0]->dtype_;
            expr in0 = in[0], in1 = in[1];
            if (in[0]->dtype_ != out_dtype) {
                in0 = builder::make_cast(out_dtype, in[0]);
            }
            if (in[1]->dtype_ != out_dtype) {
                in1 = builder::make_cast(out_dtype, in[1]);
            }
            switch (elt_op_) {
                case elt_operator::ADD:
                    return builder::make_assign_unattached(out[0], in0 + in1);
                case elt_operator::SUB:
                    return builder::make_assign_unattached(out[0], in0 - in1);
                case elt_operator::MUL:
                    return builder::make_assign_unattached(out[0], in0 * in1);
                case elt_operator::DIV:
                    return builder::make_assign_unattached(out[0],
                            make_select_by_mask(
                                    in0 / in1, mask_count, vx_info_.lanes));
                case elt_operator::MIN:
                    return builder::make_assign_unattached(
                            out[0], builder::make_min(in0, in1));
                case elt_operator::MAX:
                    return builder::make_assign_unattached(
                            out[0], builder::make_max(in0, in1));
                case elt_operator::SQD_DIFF:
                    return builder::make_assign_unattached(
                            out[0], (in0 - in1) * (in0 - in1));
                default:
                    COMPILE_ASSERT(false,
                            "Unsupport elementwise op "
                            "found.\n");
                    return stmt();
            }
        };
        // todo: currently we only support mask for div.
        bool use_mask = elt_op_ == elt_operator::DIV;
        compute_vectorized_op(inputs, *dst[0], info_, vx_info_,
                mask_compute_func_t(func), mask_compute_func_t(func), attrs_,
                wkld, use_mask);
    }
}

OP_REGISTER(add_op_t, add)
OP_REGISTER(mul_op_t, mul)
OP_REGISTER(sub_op_t, sub)
OP_REGISTER(div_op_t, div)
OP_REGISTER(min_op_t, min)
OP_REGISTER(max_op_t, max)
OP_REGISTER(squared_diff_op_t, squared_diff)

} // namespace sc
