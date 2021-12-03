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

#include <fstream>
#include <memory>
#include <vector>
#include "quantize_op.hpp"
#include <compiler/ir/graph/graph.hpp>
#include <util/math_utils.hpp>
#include <util/utils.hpp>
namespace sc {
namespace quantize {

quantize_infos_t get_quantize_info_from_attrs(const any_map_t &attrs) {
    quantize_infos_t infos;
    infos.dtype_ = attrs.get_or_else("dtype", sc_data_type_t::u8(1));
    infos.scales_ = attrs.get_or_else("scales", std::vector<float> {1.f});
    infos.zero_points_ = attrs.get_or_else("zero_points", std::vector<int> {0});
    infos.per_channel_ = attrs.get_or_else("per_channel", false)
            || infos.scales_.size() > 1;
    infos.channel_axis_ = attrs.get_or_else("channel_axis", 0);
    infos.asymmetric_ = attrs.get_or_else("asymmetric", true);
    infos.dynamic_ = attrs.get_or_else("dynamic", false);
    assert(utils::is_one_of(infos.dtype_, datatypes::f32, datatypes::bf16,
                   datatypes::u8, datatypes::s8)
            && ((infos.per_channel_ && !infos.scales_.empty())
                    || (!infos.per_channel_ && infos.scales_.size() == 1
                            && infos.zero_points_.size() <= 1))
            && (infos.asymmetric_
                    || (!infos.asymmetric_
                            && (infos.zero_points_.empty()
                                    || (infos.zero_points_.size() == 1
                                            && infos.zero_points_[0] == 0)))));
    return infos;
}

quantize_op_t::quantize_op_t(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs) {
    assert(ins.size() == 1);
    assert(ins[0]->details_.dtype_.type_code_ == sc_data_etype::F32
            || ins[0]->details_.dtype_.type_code_ == sc_data_etype::S32);
    info_.inputs_ = ins;
    if (outs.empty()) {
        // fixme: correctly infer the shape for broadcast
        info_.outputs_.emplace_back(std::make_shared<graph_tensor>(this));
        info_.outputs_[0]->details_ = ins[0]->details_;
        assert(attrs.has_key("dtype"));
        info_.outputs_[0]->details_.dtype_ = attrs.get<sc_data_type_t>("dtype");
    } else {
        info_.outputs_ = outs;
    }
    attrs_ = attrs;
    op_name_ = "quantize";
}

quantize_op_t::quantize_op_t(
        const std::vector<graph_tensor_ptr> &ins, const any_map_t &attrs)
    : quantize_op_t(ins, std::vector<graph_tensor_ptr>(), attrs) {}

std::shared_ptr<sc_graph_t> quantize_op_t::get_graph() {
    auto graph = std::make_shared<sc_graph_t>();
    // create new input logical tensors
    std::vector<graph_tensor_ptr> inputs, outputs;
    inputs = remake_logical_tensors(info_.inputs_);
    outputs = remake_logical_tensors(info_.outputs_);
    const auto qinfos = get_quantize_info_from_attrs(attrs_);
    if (qinfos.dtype_.is_etype(sc_data_etype::BF16)) {
        auto bf16_cast = graph->make("cast", inputs, {},
                {{"dtype",
                        sc_data_type_t::bf16(
                                inputs[0]->details_.dtype_.lanes_)}});
        graph->make_output(bf16_cast->get_outputs());
    } else {
        auto scales = qinfos.scales_;
        auto zeropoints = qinfos.zero_points_;
        std::vector<float> zeropoints_f32(zeropoints.begin(), zeropoints.end());
        scales = math_utils::vector_rcp(scales);
        std::shared_ptr<static_data_t> scales_ptr
                = std::make_shared<static_data_t>(scales);
        std::shared_ptr<static_data_t> zeropoints_ptr
                = std::make_shared<static_data_t>(zeropoints_f32);

        sc_dims plain_dims = {1};
        if (scales.size() > 1) {
            plain_dims.resize(inputs[0]->details_.get_plain_dims().size(), 1);
            plain_dims[qinfos.channel_axis_] = static_cast<int>(scales.size());
        }
        auto quantize_const_scales = graph->make("constant", {}, {},
                {{"values", scales_ptr}, {"dtype", datatypes::f32},
                        {"plain_dims", plain_dims},
                        {"format", sc_data_format_t()}});
        auto quantize_const_zeropoints = graph->make("constant", {}, {},
                {{"values", zeropoints_ptr}, {"dtype", datatypes::f32},
                        {"plain_dims", plain_dims},
                        {"format", sc_data_format_t()}});
        auto div_scale = graph->make("mul",
                {inputs[0], quantize_const_scales->get_outputs()[0]}, {}, {});
        if (!qinfos.zero_points_.empty()
                || (qinfos.zero_points_.size() == 1
                        && qinfos.zero_points_[0])) {
            div_scale = graph->make("add",
                    {div_scale->get_outputs()[0],
                            quantize_const_zeropoints->get_outputs()[0]},
                    {}, {});
        }
        // maybe we need clip op in future
#if 0
        auto clip = graph->make("clip", sub_zp->get_outputs(), {},
                {{"clip_min",
                         qinfos.dtype_.is_etype(sc_data_etype::U8)
                                 ? 0.f
                                 : -128.f},
                        {"clip_max",
                                qinfos.dtype_.is_etype(
                                        sc_data_etype::U8)
                                        ? 255.f
                                        : 127.f}});
#endif
        auto int8_cast = graph->make("cast", div_scale->get_outputs(), {},
                {{"dtype", qinfos.dtype_}, {"saturated", true}});
        graph->make_output(int8_cast->get_outputs());
    }
    return graph;
}

void quantize_op_t::query_format(context_ptr ctx,
        std::vector<std::vector<sc_data_format_t>> &in_formats,
        std::vector<std::vector<sc_data_format_t>> &out_formats) {
    out_formats.push_back({info_.inputs_[0]->details_.get_format()});
}

dequantize_op_t::dequantize_op_t(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs) {
    assert(ins.size() == 1);
    assert(ins[0]->details_.dtype_.type_code_ == sc_data_etype::BF16
            || ins[0]->details_.dtype_.type_code_ == sc_data_etype::U8
            || ins[0]->details_.dtype_.type_code_ == sc_data_etype::S8
            || ins[0]->details_.dtype_.type_code_ == sc_data_etype::S32);
    info_.inputs_ = ins;
    if (outs.empty()) {
        // fixme: correctly infer the shape for broadcast
        info_.outputs_.emplace_back(std::make_shared<graph_tensor>(this));
        info_.outputs_[0]->details_ = ins[0]->details_;
        info_.outputs_[0]->details_.dtype_.type_code_ = sc_data_etype::F32;
    } else {
        info_.outputs_ = outs;
    }
    attrs_ = attrs;
    op_name_ = "dequantize";
}

dequantize_op_t::dequantize_op_t(
        const std::vector<graph_tensor_ptr> &ins, const any_map_t &attrs)
    : dequantize_op_t(ins, std::vector<graph_tensor_ptr>(), attrs) {}

std::shared_ptr<sc_graph_t> dequantize_op_t::get_graph() {
    auto graph = std::make_shared<sc_graph_t>();
    // create new input logical tensors
    std::vector<graph_tensor_ptr> inputs, outputs;
    inputs = remake_logical_tensors(info_.inputs_);
    outputs = remake_logical_tensors(info_.outputs_);
    const auto qinfos = get_quantize_info_from_attrs(attrs_);
    if (qinfos.dtype_.is_etype(sc_data_etype::BF16)) {
        auto f32_cast = graph->make("cast", inputs, {},
                {{"dtype",
                        sc_data_type_t::f32(
                                outputs[0]->details_.dtype_.lanes_)}});
        graph->make_output(f32_cast->get_outputs());
    } else {
        std::vector<float> scales = qinfos.scales_;
        std::vector<float> zero_points(
                qinfos.zero_points_.begin(), qinfos.zero_points_.end());
        bool all_zero = std::all_of(qinfos.zero_points_.begin(),
                qinfos.zero_points_.end(), [](int x) { return x == 0; });
        std::shared_ptr<static_data_t> scales_ptr
                = std::make_shared<static_data_t>(scales);
        sc_dims scales_plain_dims = {1};
        if (scales.size() > 1) {
            scales_plain_dims.resize(
                    inputs[0]->details_.get_plain_dims().size(), 1);
            scales_plain_dims[qinfos.channel_axis_]
                    = static_cast<int>(scales.size());
        }
        auto const_scales = graph->make("constant", {}, {},
                {{"values", scales_ptr}, {"dtype", datatypes::f32},
                        {"plain_dims", scales_plain_dims},
                        {"format", sc_data_format_t()}});
        auto f32_cast
                = graph->make("cast", inputs, {}, {{"dtype", qinfos.dtype_}});
        if (!all_zero) {
            auto const_zero_points = graph->make("constant", {}, {},
                    {{"values", std::make_shared<static_data_t>(zero_points)},
                            {"dtype", datatypes::f32},
                            {"plain_dims", sc_dims {1}},
                            {"format", sc_data_format_t()}});
            f32_cast = graph->make("sub",
                    {f32_cast->get_outputs()[0],
                            const_zero_points->get_outputs()[0]},
                    {}, {});
        }
        auto mul_scale = graph->make("mul",
                {f32_cast->get_outputs()[0], const_scales->get_outputs()[0]},
                {}, {});
        graph->make_output(mul_scale->get_outputs());
    }
    return graph;
} // namespace quantize

void dequantize_op_t::query_format(context_ptr ctx,
        std::vector<std::vector<sc_data_format_t>> &in_formats,
        std::vector<std::vector<sc_data_format_t>> &out_formats) {
    out_formats.push_back({info_.inputs_[0]->details_.get_format()});
}
} // namespace quantize

OP_REGISTER(quantize::quantize_op_t, quantize)
OP_REGISTER(quantize::dequantize_op_t, dequantize)
} // namespace sc
