/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 23, 2022.
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
#include "Types.h"
#include <wtf/OptionSet.h>

namespace WGSL {
class BoundsCheckVisitor;
class RewriteGlobalVariables;
class TypeChecker;

namespace AST {

// A CallExpression expresses a "function" call, which consists of a target to be called,
// and a list of arguments. The target does not necesserily have to be a function identifier,
// but can also be a type, in which the whole call is a type conversion expression. The exact
// kind of expression can only be resolved during semantic analysis.
class CallExpression final : public Expression {
    WGSL_AST_BUILDER_NODE(CallExpression);

    friend BoundsCheckVisitor;
    friend RewriteGlobalVariables;
    friend TypeChecker;

public:
    using Ref = std::reference_wrapper<CallExpression>;

    NodeKind kind() const override;
    Expression& target() { return m_target.get(); }
    Expression::List& arguments() { return m_arguments; }

    bool isConstructor() const { return m_isConstructor; }

    bool isFloatToIntConversion(const Type* result = nullptr) const
    {
        const auto& getPrimitiveType = [&](const Type* type) -> const Types::Primitive* {
            if (auto* reference = std::get_if<Types::Reference>(type))
                type = reference->element;
            if (auto* vector = std::get_if<Types::Vector>(type))
                type = vector->element;
            return std::get_if<Types::Primitive>(type);
        };

        if (!m_isConstructor)
            return false;

        auto* resultPrimitive = getPrimitiveType(result ?: m_inferredType);
        if (!resultPrimitive)
            return false;
        if (resultPrimitive->kind != Types::Primitive::I32 && resultPrimitive->kind != Types::Primitive::U32)
            return false;

        if (m_arguments.size() != 1)
            return false;

        auto& argument = m_arguments[0];
        auto* argumentPrimitive = getPrimitiveType(argument.inferredType());
        if (!argumentPrimitive)
            return false;
        if (argumentPrimitive->kind != Types::Primitive::F32 && argumentPrimitive->kind != Types::Primitive::F16 && argumentPrimitive->kind != Types::Primitive::AbstractFloat)
            return false;

        return true;
    }

    const OptionSet<ShaderStage>& visibility() const { return m_visibility; }

private:
    CallExpression(SourceSpan span, Expression::Ref&& target, Expression::List&& arguments)
        : Expression(span)
        , m_target(WTFMove(target))
        , m_arguments(WTFMove(arguments))
    { }

    // If m_target is a NamedType, it could either be a:
    //   * Type that does not accept parameters (bool, i32, u32, ...)
    //   * Identifier that refers to a type alias.
    //   * Identifier that refers to a function.
    Expression::Ref m_target;
    Expression::List m_arguments;

    bool m_isConstructor { false };
    OptionSet<ShaderStage> m_visibility { ShaderStage::Compute, ShaderStage::Vertex, ShaderStage::Fragment };
};

} // namespace AST
} // namespace WGSL

SPECIALIZE_TYPE_TRAITS_WGSL_AST(CallExpression)
