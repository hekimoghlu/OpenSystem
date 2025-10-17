/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 8, 2022.
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

#include "ASTAttribute.h"
#include "ASTCompoundStatement.h"
#include "ASTExpression.h"

namespace WGSL::AST {

class IfStatement final : public Statement {
    WGSL_AST_BUILDER_NODE(IfStatement);
public:
    NodeKind kind() const override;
    Expression& test() { return m_test.get(); }
    CompoundStatement& trueBody() { return m_trueBody.get(); }
    Statement* maybeFalseBody() { return m_falseBody; }
    Attribute::List& attributes() { return m_attributes; }

private:
    IfStatement(SourceSpan span, Expression::Ref&& test, CompoundStatement::Ref&& trueBody, Statement::Ptr falseBody, Attribute::List&& attributes)
        : Statement(span)
        , m_test(WTFMove(test))
        , m_trueBody(WTFMove(trueBody))
        , m_falseBody(falseBody)
        , m_attributes(WTFMove(attributes))
    { }

    Expression::Ref m_test;
    CompoundStatement::Ref m_trueBody;
    Statement::Ptr m_falseBody;
    Attribute::List m_attributes;
};

} // namespace WGSL::AST

SPECIALIZE_TYPE_TRAITS_WGSL_AST(IfStatement)
