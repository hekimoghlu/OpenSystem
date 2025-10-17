/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 5, 2023.
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
#include <array>
#include <wtf/EnumTraits.h>
#include <wtf/Forward.h>
#include <wtf/text/ASCIILiteral.h>

#define WGSL_AST_UNARYOP_IMPL \
    WGSL_AST_UNARYOP(AddressOf, "&") \
    WGSL_AST_UNARYOP(Complement, "~") \
    WGSL_AST_UNARYOP(Dereference, "*") \
    WGSL_AST_UNARYOP(Negate, "-") \
    WGSL_AST_UNARYOP(Not, "!")

namespace WGSL::AST {

enum class UnaryOperation : uint8_t {
#define WGSL_AST_UNARYOP(x, y) x,
WGSL_AST_UNARYOP_IMPL
#undef WGSL_AST_UNARYOP
};

constexpr ASCIILiteral toASCIILiteral(UnaryOperation op)
{
    constexpr auto unaryOperationNames = std::to_array<ASCIILiteral>({
#define WGSL_AST_UNARYOP(x, y) y##_s,
WGSL_AST_UNARYOP_IMPL
#undef WGSL_AST_UNARYOP
    });

    return unaryOperationNames[WTF::enumToUnderlyingType(op)];
}

void printInternal(PrintStream&, UnaryOperation);

class UnaryExpression final : public Expression {
    WGSL_AST_BUILDER_NODE(UnaryExpression);
public:
    NodeKind kind() const final;
    Expression& expression() { return m_expression.get(); }
    UnaryOperation operation() const { return m_operation; }

private:
    UnaryExpression(SourceSpan span, Expression::Ref&& expression, UnaryOperation operation)
        : Expression(span)
        , m_expression(WTFMove(expression))
        , m_operation(operation)
    { }

    Expression::Ref m_expression;
    UnaryOperation m_operation;
};


} // namespace WGSL::AST

SPECIALIZE_TYPE_TRAITS_WGSL_AST(UnaryExpression)
