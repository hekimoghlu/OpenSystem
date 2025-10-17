/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 12, 2024.
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

#include "ASTExpression.h"
#include "ASTStatement.h"

namespace WGSL::AST {

class DecrementIncrementStatement final : public Statement {
    WGSL_AST_BUILDER_NODE(DecrementIncrementStatement);
public:
    enum class Operation : uint8_t {
        Decrement,
        Increment,
    };

    NodeKind kind() const override;
    Expression& expression() { return m_expression; }
    Operation operation() const { return m_operation; }

private:
    DecrementIncrementStatement(SourceSpan span, Expression::Ref&& expression, Operation operation)
        : Statement(span)
        , m_expression(WTFMove(expression))
        , m_operation(operation)
    { }

    Expression::Ref m_expression;
    Operation m_operation;
};

void printInternal(PrintStream&, DecrementIncrementStatement::Operation);

} // namespace WGSL::AST

SPECIALIZE_TYPE_TRAITS_WGSL_AST(DecrementIncrementStatement)
