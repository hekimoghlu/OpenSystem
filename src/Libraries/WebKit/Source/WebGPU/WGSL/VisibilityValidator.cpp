/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 5, 2024.
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
#include "VisibilityValidator.h"

#include "AST.h"
#include "ASTVisitor.h"
#include "WGSLEnums.h"
#include "WGSLShaderModule.h"
#include <wtf/text/MakeString.h>

namespace WGSL {

class VisibilityValidator : public AST::Visitor {
public:
    VisibilityValidator(ShaderModule&);

    std::optional<FailedCheck> validate();

    void visit(AST::Function&) override;
    void visit(AST::CallExpression&) override;

private:
    template<typename... Arguments>
    void error(const SourceSpan&, Arguments&&...);

    ShaderModule& m_shaderModule;
    Vector<Error> m_errors;
    ShaderStage m_stage;
};

VisibilityValidator::VisibilityValidator(ShaderModule& shaderModule)
    : m_shaderModule(shaderModule)
{
}

std::optional<FailedCheck> VisibilityValidator::validate()
{
    for (auto& entryPoint : m_shaderModule.callGraph().entrypoints()) {
        m_stage = entryPoint.stage;
        visit(entryPoint.function);

    }

    if (m_errors.isEmpty())
        return std::nullopt;
    return FailedCheck { WTFMove(m_errors), { } };
}

void VisibilityValidator::visit(AST::Function& function)
{
    AST::Visitor::visit(function);

    for (auto& callee : m_shaderModule.callGraph().callees(function))
        visit(*callee.target);
}

void VisibilityValidator::visit(AST::CallExpression& call)
{
    AST::Visitor::visit(call);
    if (!call.visibility().contains(m_stage))
        error(call.span(), "built-in cannot be used by "_s, toString(m_stage), " pipeline stage"_s);
}

template<typename... Arguments>
void VisibilityValidator::error(const SourceSpan& span, Arguments&&... arguments)
{
    m_errors.append({ makeString(std::forward<Arguments>(arguments)...), span });
}

std::optional<FailedCheck> validateVisibility(ShaderModule& shaderModule)
{
    return VisibilityValidator(shaderModule).validate();
}

} // namespace WGSL
