/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 18, 2024.
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
#include "SourceSpan.h"
#include <wtf/TypeCasts.h>

namespace WGSL::AST {

enum class NodeKind : uint8_t {
    Unknown,

    // Attribute
    AlignAttribute,
    BindingAttribute,
    BuiltinAttribute,
    ConstAttribute,
    DiagnosticAttribute,
    GroupAttribute,
    IdAttribute,
    InterpolateAttribute,
    InvariantAttribute,
    LocationAttribute,
    MustUseAttribute,
    SizeAttribute,
    StageAttribute,
    WorkgroupSizeAttribute,

    ConstAssert,
    Directive,
    DiagnosticDirective,

    // Expression
    BinaryExpression,
    BitcastExpression,
    CallExpression,
    FieldAccessExpression,
    IdentifierExpression,
    IdentityExpression,
    IndexAccessExpression,
    PointerDereferenceExpression,
    UnaryExpression,

    Function,
    Parameter,

    Identifier,

    // Literal
    AbstractFloatLiteral,
    AbstractIntegerLiteral,
    BoolLiteral,
    Float32Literal,
    Float16Literal,
    Signed32Literal,
    Unsigned32Literal,

    ShaderModule,

    // Statement
    AssignmentStatement,
    BreakStatement,
    CallStatement,
    CompoundAssignmentStatement,
    CompoundStatement,
    ConstAssertStatement,
    ContinueStatement,
    DecrementIncrementStatement,
    DiscardStatement,
    ForStatement,
    IfStatement,
    LoopStatement,
    PhonyAssignmentStatement,
    ReturnStatement,
    StaticAssertStatement,
    SwitchStatement,
    VariableStatement,
    WhileStatement,

    Structure,
    StructureMember,

    TypeAlias,

    ArrayTypeExpression,
    ElaboratedTypeExpression,
    ReferenceTypeExpression,

    Variable,

    VariableQualifier
};

class Node {
    WGSL_AST_BUILDER_NODE(Node);
public:
    virtual ~Node() = default;

    virtual NodeKind kind() const { return NodeKind::Unknown; };
    const SourceSpan& span() const { return m_span; }

protected:
    Node(SourceSpan span)
        : m_span(span)
    { }

private:
    SourceSpan m_span;
};

} // namespace WGSL::AST

#define SPECIALIZE_TYPE_TRAITS_WGSL_AST(Kind) \
inline WGSL::AST::NodeKind WGSL::AST::Kind::kind() const { return WGSL::AST::NodeKind::Kind; } \
SPECIALIZE_TYPE_TRAITS_BEGIN(WGSL::AST::Kind)                           \
static bool isType(const WGSL::AST::Node& node) { return node.kind() == WGSL::AST::NodeKind::Kind; } \
SPECIALIZE_TYPE_TRAITS_END()
