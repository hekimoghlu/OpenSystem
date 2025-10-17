/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 14, 2021.
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
#include "config.h"
#include "BoundsCheck.h"

#include "AST.h"
#include "ASTVisitor.h"
#include "Types.h"
#include "WGSLShaderModule.h"

namespace WGSL {

class BoundsCheckVisitor : AST::Visitor {
public:
    BoundsCheckVisitor(ShaderModule& shaderModule)
        : m_shaderModule(shaderModule)
    {
    }

    std::optional<FailedCheck> run()
    {
        AST::Visitor::visit(m_shaderModule);
        return std::nullopt;
    }

    void visit(AST::IndexAccessExpression&) override;

private:
    ShaderModule& m_shaderModule;
};


void BoundsCheckVisitor::visit(AST::IndexAccessExpression& access)
{
    if (access.constantValue())
        return;

    AST::Visitor::visit(access);

    const auto& constant = [&](unsigned size) -> AST::Expression& {
        auto& sizeExpression =  m_shaderModule.astBuilder().construct<AST::Unsigned32Literal>(
            SourceSpan::empty(),
            size
        );
        sizeExpression.m_inferredType = m_shaderModule.types().u32Type();
        sizeExpression.setConstantValue(size);
        return sizeExpression;
    };

    const auto& replace = [&](AST::Expression& size) {
        auto* index = &access.index();
        if (index->inferredType() != m_shaderModule.types().u32Type()) {
            auto& u32Target = m_shaderModule.astBuilder().construct<AST::IdentifierExpression>(
                SourceSpan::empty(),
                AST::Identifier::make("u32"_s)
            );
            u32Target.m_inferredType = m_shaderModule.types().bottomType();

            auto& u32Call = m_shaderModule.astBuilder().construct<AST::CallExpression>(
                SourceSpan::empty(),
                u32Target,
                AST::Expression::List { *index }
            );
            u32Call.m_inferredType = m_shaderModule.types().u32Type();
            u32Call.m_isConstructor = true;
            index = &u32Call;
        }

        auto& minTarget = m_shaderModule.astBuilder().construct<AST::IdentifierExpression>(
            SourceSpan::empty(),
            AST::Identifier::make("__wgslMin"_s)
        );
        minTarget.m_inferredType = m_shaderModule.types().bottomType();

        auto& one =  m_shaderModule.astBuilder().construct<AST::Unsigned32Literal>(
            SourceSpan::empty(),
            1
        );
        one.m_inferredType = m_shaderModule.types().u32Type();
        one.setConstantValue(1u);

        auto& upperBound = m_shaderModule.astBuilder().construct<AST::BinaryExpression>(
            SourceSpan::empty(),
            size,
            one,
            AST::BinaryOperation::Subtract
        );
        upperBound.m_inferredType = m_shaderModule.types().u32Type();

        auto& minCall = m_shaderModule.astBuilder().construct<AST::CallExpression>(
            SourceSpan::empty(),
            minTarget,
            AST::Expression::List { *index, upperBound }
        );
        minCall.m_inferredType = upperBound.inferredType();

        auto& newAccess = m_shaderModule.astBuilder().construct<AST::IndexAccessExpression>(
            access.span(),
            access.base(),
            minCall
        );
        newAccess.m_inferredType = access.inferredType();

        m_shaderModule.replace(access, newAccess);
        m_shaderModule.setUsesMin();
    };

    auto* base = access.base().inferredType();
    if (auto* reference = std::get_if<Types::Reference>(base))
        base = reference->element;
    if (auto* pointer = std::get_if<Types::Pointer>(base))
        base = pointer->element;

    if (auto* vector = std::get_if<Types::Vector>(base)) {
        replace(constant(vector->size));
        return;
    }

    if (auto* matrix = std::get_if<Types::Matrix>(base)) {
        replace(constant(matrix->columns));
        return;
    }

    auto& array = std::get<Types::Array>(*base);
    WTF::switchOn(array.size,
        [&](unsigned size) {
            replace(constant(size));
        },
        [&](AST::Expression* size) {
            // FIXME: this should be a pipeline creation error, not a runtime error
            replace(*size);
        },
        [&](std::monostate) {
            auto& target = m_shaderModule.astBuilder().construct<AST::IdentifierExpression>(
                SourceSpan::empty(),
                AST::Identifier::make("arrayLength"_s)
            );
            target.m_inferredType = m_shaderModule.types().bottomType();


            auto* argument = &access.base();
            if (auto* reference = std::get_if<Types::Reference>(access.base().inferredType())) {
                auto& addressOf = m_shaderModule.astBuilder().construct<AST::UnaryExpression>(
                    SourceSpan::empty(),
                    access.base(),
                    AST::UnaryOperation::AddressOf
                );
                addressOf.m_inferredType = m_shaderModule.types().pointerType(
                    reference->addressSpace,
                    reference->element,
                    reference->accessMode
                );
                argument = &addressOf;
            }

            RELEASE_ASSERT(std::holds_alternative<Types::Pointer>(*argument->inferredType()));
            auto& call = m_shaderModule.astBuilder().construct<AST::CallExpression>(
                SourceSpan::empty(),
                target,
                AST::Expression::List { *argument }
            );
            call.m_inferredType = m_shaderModule.types().u32Type();

            replace(call);
        });
}

std::optional<FailedCheck> insertBoundsChecks(ShaderModule& shaderModule)
{
    return BoundsCheckVisitor(shaderModule).run();
}

} // namespace WGSL
