/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 21, 2025.
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

namespace WGSL::AST {

#define WGSL_AST_BINOP_IMPL \
    WGSL_AST_BINOP(Add, "+") \
    WGSL_AST_BINOP(Subtract, "-") \
    WGSL_AST_BINOP(Multiply, "*") \
    WGSL_AST_BINOP(Divide, "/") \
    WGSL_AST_BINOP(Modulo, "%") \
    WGSL_AST_BINOP(And, "&") \
    WGSL_AST_BINOP(Or, "|") \
    WGSL_AST_BINOP(Xor, "^") \
    WGSL_AST_BINOP(LeftShift, "<<") \
    WGSL_AST_BINOP(RightShift, ">>") \
    WGSL_AST_BINOP(Equal, "==") \
    WGSL_AST_BINOP(NotEqual, "!=") \
    WGSL_AST_BINOP(GreaterThan, ">") \
    WGSL_AST_BINOP(GreaterEqual, ">=") \
    WGSL_AST_BINOP(LessThan, "<") \
    WGSL_AST_BINOP(LessEqual, "<=") \
    WGSL_AST_BINOP(ShortCircuitAnd, "&&") \
    WGSL_AST_BINOP(ShortCircuitOr, "||")

enum class BinaryOperation : uint8_t {
#define WGSL_AST_BINOP(x, y) x,
    WGSL_AST_BINOP_IMPL
#undef WGSL_AST_BINOP
};

constexpr ASCIILiteral toASCIILiteral(BinaryOperation op)
{
    constexpr auto binaryOperationNames = std::to_array<ASCIILiteral>({
#define WGSL_AST_BINOP(x, y) y##_s,
        WGSL_AST_BINOP_IMPL
#undef WGSL_AST_BINOP
    });

    return binaryOperationNames[WTF::enumToUnderlyingType(op)];
}

void printInternal(PrintStream&, BinaryOperation);

class BinaryExpression : public Expression {
    WGSL_AST_BUILDER_NODE(BinaryExpression);
public:
    NodeKind kind() const override;
    BinaryOperation operation() const { return m_operation; }
    Expression& leftExpression() { return m_lhs.get(); }
    Expression& rightExpression() { return m_rhs.get(); }

private:
    BinaryExpression(SourceSpan span, Expression::Ref&& lhs, Expression::Ref&& rhs, BinaryOperation operation)
        : Expression(span)
        , m_lhs(WTFMove(lhs))
        , m_rhs(WTFMove(rhs))
        , m_operation(operation)
    { }

    Expression::Ref m_lhs;
    Expression::Ref m_rhs;
    BinaryOperation m_operation;
};

} // namespace WGSL::AST

SPECIALIZE_TYPE_TRAITS_WGSL_AST(BinaryExpression)
