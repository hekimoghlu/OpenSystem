/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 1, 2022.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#pragma once

#include "ASTCompoundStatement.h"
#include "ASTExpression.h"

namespace WGSL::AST {

class WhileStatement final : public Statement {
    WGSL_AST_BUILDER_NODE(WhileStatement);
public:
    NodeKind kind() const final;
    Expression& test() { return m_test.get(); }
    CompoundStatement& body() { return m_body.get(); }

private:
    WhileStatement(SourceSpan span, Expression::Ref&& test, CompoundStatement::Ref&& body)
        : Statement(span)
        , m_test(WTFMove(test))
        , m_body(WTFMove(body))
    { }

    Expression::Ref m_test;
    CompoundStatement::Ref m_body;
};

} // namespace WGSL::AST

SPECIALIZE_TYPE_TRAITS_WGSL_AST(WhileStatement)
