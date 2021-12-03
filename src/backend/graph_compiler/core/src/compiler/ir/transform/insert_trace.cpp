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
#include "insert_trace.hpp"
#include <atomic>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <compiler/ir/builder.hpp>
#include <compiler/ir/visitor.hpp>
#include <microkernel/builtin.hpp>

namespace sc {

extern int register_traced_func(const std::string &name);
class trace_inserter_impl_t : public ir_visitor_t {
public:
    using ir_visitor_t::dispatch;
    int func_id;
    func_c dispatch(func_c v) override {
        func_id = register_traced_func(v->name_);
        auto oldbody = v->body_;
        assert(oldbody.isa<stmts>());
        const auto &seq = oldbody.static_as<stmts>()->seq_;
        bool is_return_last = !seq.empty() && seq.back().isa<returns>();
        std::vector<stmt> newseq;
        newseq.emplace_back(builder::make_evaluate_unattached(
                builtin::make_trace(func_id, 0)));
        for (const auto &s : seq) {
            newseq.emplace_back(dispatch(s).remove_const());
        }
        if (!is_return_last) {
            newseq.emplace_back(builder::make_evaluate_unattached(
                    builtin::make_trace(func_id, 1)));
        }
        return copy_attr(*v,
                builder::make_func(v->name_, v->params_,
                        make_stmt<stmts_node_t>(std::move(newseq)),
                        v->ret_type_));
    }

    stmt_c visit(returns_c v) override {
        return builder::make_stmts_unattached(
                {builder::make_evaluate_unattached(
                         builtin::make_trace(func_id, 1)),
                        v});
    }
};

const_ir_module_ptr trace_inserter_t::operator()(const_ir_module_ptr m) {
    auto ret = std::make_shared<ir_module_t>(*m);
    auto &funcs = ret->get_contents();
    for (unsigned i = 0; i < funcs.size(); i++) {
        funcs[i] = std::const_pointer_cast<func_base>(
                trace_inserter_impl_t().dispatch(funcs[i]));
    }
    return ret;
}

} // namespace sc
