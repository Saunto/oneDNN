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

#include "intrinsics.hpp"
#include <memory>
#include <string>
#include <util/any_map.hpp>

namespace sc {
intrinsic_handler_t::intrinsic_handler_t(const std::string &name)
    : name_(name) {}

struct binary_intrinsic_handler_t : public intrinsic_handler_t {
    binary_intrinsic_handler_t(const std::string &name)
        : intrinsic_handler_t(name) {}
    void on_initialize(intrin_call_node &node) override;
};

void binary_intrinsic_handler_t::on_initialize(intrin_call_node &node) {
    assert(node.args_.size() == 2);
    auto &l = node.args_[0];
    auto &r = node.args_[1];
    node.dtype_ = l->dtype_ == r->dtype_ ? l->dtype_ : datatypes::undef;
}

struct trinary_intrinsic_handler_t : public intrinsic_handler_t {
    trinary_intrinsic_handler_t(const std::string &name)
        : intrinsic_handler_t(name) {}
    void on_initialize(intrin_call_node &node) override;
};

void trinary_intrinsic_handler_t::on_initialize(intrin_call_node &node) {
    assert(node.args_.size() == 3);
    auto &a = node.args_[0];
    auto &b = node.args_[1];
    auto &c = node.args_[2];
    if (node.type_ == intrin_type::permute
            || node.type_ == intrin_type::shuffle) {
        node.dtype_ = a->dtype_ == b->dtype_ ? a->dtype_ : datatypes::undef;
    } else {
        node.dtype_ = a->dtype_ == b->dtype_ && a->dtype_ == c->dtype_
                ? a->dtype_
                : datatypes::undef;
    }
}

struct min_handler_t : public binary_intrinsic_handler_t {
    min_handler_t() : binary_intrinsic_handler_t("min") {}
};

struct max_handler_t : public binary_intrinsic_handler_t {
    max_handler_t() : binary_intrinsic_handler_t("max") {}
};

struct abs_handler_t : public intrinsic_handler_t {
    void on_initialize(intrin_call_node &node) override {
        assert(node.args_.size() == 1);
        node.dtype_ = node.args_[0]->dtype_;
    }
    abs_handler_t() : intrinsic_handler_t("abs") {}
};

struct round_handler_t : public intrinsic_handler_t {
    void on_initialize(intrin_call_node &node) override {
        assert(node.args_.size() == 1);
        node.dtype_ = node.args_[0]->dtype_;
    }
    round_handler_t() : intrinsic_handler_t("round") {}
};

struct ceil_handler_t : public intrinsic_handler_t {
    void on_initialize(intrin_call_node &node) override {
        assert(node.args_.size() == 1);
        node.dtype_ = node.args_[0]->dtype_;
    }
    ceil_handler_t() : intrinsic_handler_t("ceil") {}
};

struct floor_handler_t : public intrinsic_handler_t {
    void on_initialize(intrin_call_node &node) override {
        assert(node.args_.size() == 1);
        node.dtype_ = node.args_[0]->dtype_;
    }
    floor_handler_t() : intrinsic_handler_t("floor") {}
};

struct exp_handler_t : public intrinsic_handler_t {
    void on_initialize(intrin_call_node &node) override {
        assert(node.args_.size() == 1);
        node.dtype_ = node.args_[0]->dtype_;
    }
    exp_handler_t() : intrinsic_handler_t("exp") {}
};

struct sqrt_handler_t : public intrinsic_handler_t {
    void on_initialize(intrin_call_node &node) override {
        assert(node.args_.size() == 1);
        node.dtype_ = node.args_[0]->dtype_;
    }
    sqrt_handler_t() : intrinsic_handler_t("sqrt") {}
};

struct rsqrt_handler_t : public intrinsic_handler_t {
    void on_initialize(intrin_call_node &node) override {
        assert(node.args_.size() == 1);
        node.dtype_ = node.args_[0]->dtype_;
    }
    rsqrt_handler_t() : intrinsic_handler_t("rsqrt") {}
};

struct reduce_add_handler_t : public intrinsic_handler_t {
    void on_initialize(intrin_call_node &node) override {
        assert(node.args_.size() == 1);
        node.dtype_ = node.args_[0]->dtype_;
        node.dtype_.lanes_ = 1;
    }
    reduce_add_handler_t() : intrinsic_handler_t("reduce_add") {}
};

struct reduce_mul_handler_t : public intrinsic_handler_t {
    void on_initialize(intrin_call_node &node) override {
        assert(node.args_.size() == 1);
        node.dtype_ = node.args_[0]->dtype_;
        node.dtype_.lanes_ = 1;
    }
    reduce_mul_handler_t() : intrinsic_handler_t("reduce_mul") {}
};

struct broadcast_handler_t : public intrinsic_handler_t {
    void on_initialize(intrin_call_node &node) override {
        assert(node.args_.size() == 2);
        COMPILE_ASSERT(
                node.args_[1].isa<constant>(), "Expecting constant node");
        auto lanes = get_const_as_int(node.args_[1].static_as<constant>());
        COMPILE_ASSERT(lanes <= 512, "Expecting lanes<=512");
        node.dtype_ = node.args_[0]->dtype_;
        node.dtype_.lanes_ = lanes;
    }
    broadcast_handler_t() : intrinsic_handler_t("broadcast") {}
};

struct fmadd_handler_t : public trinary_intrinsic_handler_t {
    void on_initialize(intrin_call_node &node) override {
        node.dtype_ = node.args_[0]->dtype_;
    }
    fmadd_handler_t() : trinary_intrinsic_handler_t("fmadd") {}
};

struct unpack_low_handler_t : public binary_intrinsic_handler_t {
    unpack_low_handler_t() : binary_intrinsic_handler_t("unpack_low") {}
};

struct unpack_high_handler_t : public binary_intrinsic_handler_t {
    unpack_high_handler_t() : binary_intrinsic_handler_t("unpack_high") {}
};

struct shuffle_handler_t : public trinary_intrinsic_handler_t {
    shuffle_handler_t() : trinary_intrinsic_handler_t("shuffle") {}
};

struct permute_handler_t : public trinary_intrinsic_handler_t {
    permute_handler_t() : trinary_intrinsic_handler_t("permute") {}
};

struct reinterpret_handler_t : public intrinsic_handler_t {
    void on_initialize(intrin_call_node &node) override {
        assert(node.args_.size() == 1);
        node.dtype_ = node.intrin_attrs_->get<sc_data_type_t>(
                intrin_attr::out_dtype);
    }
    reinterpret_handler_t() : intrinsic_handler_t("reinterpret") {}
};

struct round_and_cast_handler_t : public intrinsic_handler_t {
    void on_initialize(intrin_call_node &node) override {
        assert(node.args_.size() == 1);
        node.dtype_ = node.intrin_attrs_->get<sc_data_type_t>(
                intrin_attr::out_dtype);
        COMPILE_ASSERT(node.dtype_.lanes_ == node.args_[0]->dtype_.lanes_
                        && node.dtype_.type_code_ == sc_data_etype::S32
                        && node.args_[0]->dtype_.type_code_
                                == sc_data_etype::F32,
                "round_and_cast cannot handle " << node.args_[0]->dtype_ << "->"
                                                << node.dtype_);
    }
    round_and_cast_handler_t() : intrinsic_handler_t("round_and_cast") {}
};

struct isnan_handler_t : public intrinsic_handler_t {
    void on_initialize(intrin_call_node &node) override {
        assert(node.args_.size() == 1);
        node.dtype_ = sc_data_type_t::boolean(node.dtype_.lanes_);
    }
    isnan_handler_t() : intrinsic_handler_t("isnan") {}
};

struct saturated_cast_handler_t : public intrinsic_handler_t {
    void on_initialize(intrin_call_node &node) override {
        assert(node.args_.size() == 1);
        node.dtype_ = node.intrin_attrs_->get<sc_data_type_t>("out_dtype");
    }
    saturated_cast_handler_t() : intrinsic_handler_t("saturated_cast") {}
};

struct shl_handler_t : public intrinsic_handler_t {
    void on_initialize(intrin_call_node &node) override {
        assert(node.args_.size() == 2);
        node.dtype_ = node.args_[0]->dtype_;
    }
    shl_handler_t() : intrinsic_handler_t("shl") {}
};

struct shr_handler_t : public intrinsic_handler_t {
    void on_initialize(intrin_call_node &node) override {
        assert(node.args_.size() == 2);
        node.dtype_ = node.args_[0]->dtype_;
    }
    shr_handler_t() : intrinsic_handler_t("shr") {}
};

struct int_and_handler_t : public binary_intrinsic_handler_t {
    int_and_handler_t() : binary_intrinsic_handler_t("int_and") {}
};

struct int_or_handler_t : public binary_intrinsic_handler_t {
    int_or_handler_t() : binary_intrinsic_handler_t("int_or") {}
};

struct int_xor_handler_t : public binary_intrinsic_handler_t {
    int_xor_handler_t() : binary_intrinsic_handler_t("int_xor") {}
};

struct brgemm_handler_t : public intrinsic_handler_t {
    size_t arg_cnt_;
    void on_initialize(intrin_call_node &node) override {
        assert(node.args_.size() == arg_cnt_);
        node.dtype_ = datatypes::void_t;
    }
    brgemm_handler_t(size_t arg_cnt, const char *name)
        : intrinsic_handler_t(name), arg_cnt_(arg_cnt) {}
};

namespace brgemm_args {
sc_data_type_t arg_types[NUM_ARGS_CPU] = {
        datatypes::pointer, // A (overloaded)
        datatypes::pointer, // B
        datatypes::pointer, // C
        datatypes::s32, // num
        datatypes::s32, // M
        datatypes::s32, // N
        datatypes::s32, // K
        datatypes::s32, // LDA
        datatypes::s32, // LDB
        datatypes::s32, // LDC
        datatypes::s32, // stride_a
        datatypes::s32, // stride_b
};

sc_data_type_t list_arg_types[NUM_ARGS_LIST] = {
        datatypes::pointer, // A
        datatypes::pointer, // B
        datatypes::pointer, // C
        datatypes::s32, // num
        datatypes::s32, // M
        datatypes::s32, // N
        datatypes::s32, // K
        datatypes::s32, // LDA
        datatypes::s32, // LDB
        datatypes::s32, // LDC
        datatypes::s32, // stride_a
        datatypes::s32, // stride_b
        datatypes::s32, // len
};
} // namespace brgemm_args

static std::unique_ptr<intrinsic_handler_t> handlers[]
        = {utils::make_unique<min_handler_t>(),
                utils::make_unique<max_handler_t>(),
                utils::make_unique<abs_handler_t>(),
                utils::make_unique<round_handler_t>(),
                utils::make_unique<ceil_handler_t>(),
                utils::make_unique<floor_handler_t>(),
                utils::make_unique<exp_handler_t>(),
                utils::make_unique<sqrt_handler_t>(),
                utils::make_unique<rsqrt_handler_t>(),
                utils::make_unique<reduce_add_handler_t>(),
                utils::make_unique<reduce_mul_handler_t>(),
                utils::make_unique<fmadd_handler_t>(),
                utils::make_unique<unpack_low_handler_t>(),
                utils::make_unique<unpack_high_handler_t>(),
                utils::make_unique<shuffle_handler_t>(),
                utils::make_unique<permute_handler_t>(),
                utils::make_unique<int_and_handler_t>(),
                utils::make_unique<int_or_handler_t>(),
                utils::make_unique<int_xor_handler_t>(),
                utils::make_unique<reinterpret_handler_t>(),
                utils::make_unique<broadcast_handler_t>(),
                utils::make_unique<isnan_handler_t>(),
                utils::make_unique<saturated_cast_handler_t>(),
                utils::make_unique<round_and_cast_handler_t>(),
                utils::make_unique<shl_handler_t>(),
                utils::make_unique<shr_handler_t>(),
                utils::make_unique<brgemm_handler_t>(
                        brgemm_args::NUM_ARGS_CPU, "brgemm"),
                utils::make_unique<brgemm_handler_t>(
                        brgemm_args::NUM_ARGS_LIST, "list_brgemm")};

static_assert(sizeof(handlers) / sizeof(handlers[0])
                == int(intrin_type::NUM_INTRINSICS),
        "Not all intrinsics are filled in handlers");

intrinsic_handler_t &get_intrinsic_handler(intrin_type intrin) {
    return *handlers[static_cast<int>(intrin)];
}

} // namespace sc
