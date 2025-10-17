/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 20, 2022.
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

#include "ASTBuilder.h"
#include "ASTNode.h"
#include "ConstantValue.h"
#include <wtf/ReferenceWrapperVector.h>

namespace WGSL {
class BoundsCheckVisitor;
class ConstantRewriter;
class EntryPointRewriter;
class PointerRewriter;
class RewriteGlobalVariables;
class TypeChecker;
struct Type;

namespace AST {

class Expression : public Node {
    WGSL_AST_BUILDER_NODE(Expression);
    friend BoundsCheckVisitor;
    friend ConstantRewriter;
    friend EntryPointRewriter;
    friend PointerRewriter;
    friend RewriteGlobalVariables;
    friend TypeChecker;

public:
    using Ref = std::reference_wrapper<Expression>;
    using Ptr = Expression*;
    using List = ReferenceWrapperVector<Expression>;

    virtual ~Expression() { }

    const Type* inferredType() const { return m_inferredType; }

    const std::optional<ConstantValue>& constantValue() const { return m_constantValue; }
    void setConstantValue(ConstantValue value) { m_constantValue = value; }

protected:
    Expression(SourceSpan span)
        : Node(span)
    { }

    const Type* m_inferredType { nullptr };

private:
    std::optional<ConstantValue> m_constantValue { std::nullopt };
};

} // namespace AST
} // namespace WGSL

SPECIALIZE_TYPE_TRAITS_BEGIN(WGSL::AST::Expression)
static bool isType(const WGSL::AST::Node& node)
{
    switch (node.kind()) {
        // Expressions
    case WGSL::AST::NodeKind::BinaryExpression:
    case WGSL::AST::NodeKind::BitcastExpression:
    case WGSL::AST::NodeKind::IndexAccessExpression:
    case WGSL::AST::NodeKind::CallExpression:
    case WGSL::AST::NodeKind::IdentifierExpression:
    case WGSL::AST::NodeKind::FieldAccessExpression:
    case WGSL::AST::NodeKind::UnaryExpression:
        // Literals
    case WGSL::AST::NodeKind::AbstractFloatLiteral:
    case WGSL::AST::NodeKind::AbstractIntegerLiteral:
    case WGSL::AST::NodeKind::BoolLiteral:
    case WGSL::AST::NodeKind::Float32Literal:
    case WGSL::AST::NodeKind::Float16Literal:
    case WGSL::AST::NodeKind::Signed32Literal:
    case WGSL::AST::NodeKind::Unsigned32Literal:
        return true;
    default:
        return false;
    }
}
SPECIALIZE_TYPE_TRAITS_END()
