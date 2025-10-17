/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 18, 2025.
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
#include "ASTDeclaration.h"
#include "ASTParameter.h"
#include "ASTWorkgroupSizeAttribute.h"

#include <wtf/UniqueRefVector.h>

namespace WGSL {

class AttributeValidator;

namespace AST {

class Function final : public Declaration {
    WGSL_AST_BUILDER_NODE(Function);
    friend AttributeValidator;

public:
    NodeKind kind() const override;
    Identifier& name() override { return m_name; }
    Parameter::List& parameters() { return m_parameters; }
    Attribute::List& attributes() { return m_attributes; }
    Attribute::List& returnAttributes() { return m_returnAttributes; }
    Expression* maybeReturnType() { return m_returnType; }
    CompoundStatement& body() { return m_body.get(); }
    const Identifier& name() const { return m_name; }
    const Parameter::List& parameters() const { return m_parameters; }
    const Attribute::List& attributes() const { return m_attributes; }
    const Attribute::List& returnAttributes() const { return m_returnAttributes; }
    const Expression* maybeReturnType() const { return m_returnType; }
    const CompoundStatement& body() const { return m_body.get(); }

    bool mustUse() const { return m_mustUse; }
    std::optional<ShaderStage> stage() const { return m_stage; }
    const std::optional<WorkgroupSize>& workgroupSize() const { return m_workgroupSize; }

    bool returnTypeInvariant() const { return m_returnTypeInvariant; }
    std::optional<Builtin> returnTypeBuiltin() const { return m_returnTypeBuiltin; }
    std::optional<Interpolation> returnTypeInterpolation() const { return m_returnTypeInterpolation; }
    std::optional<unsigned> returnTypeLocation() const { return m_returnTypeLocation; }

private:
    Function(SourceSpan span, Identifier&& name, Parameter::List&& parameters, Expression::Ptr returnType, CompoundStatement::Ref&& body, Attribute::List&& attributes, Attribute::List&& returnAttributes)
        : Declaration(span)
        , m_name(WTFMove(name))
        , m_parameters(WTFMove(parameters))
        , m_attributes(WTFMove(attributes))
        , m_returnAttributes(WTFMove(returnAttributes))
        , m_returnType(returnType)
        , m_body(WTFMove(body))
    { }

    Identifier m_name;
    Parameter::List m_parameters;
    Attribute::List m_attributes;
    Attribute::List m_returnAttributes;
    Expression::Ptr m_returnType;
    CompoundStatement::Ref m_body;

    // Attributes
    bool m_mustUse { false };
    std::optional<ShaderStage> m_stage;
    std::optional<WorkgroupSize> m_workgroupSize;

    bool m_returnTypeInvariant { false };
    std::optional<Builtin> m_returnTypeBuiltin;
    std::optional<Interpolation> m_returnTypeInterpolation;
    std::optional<unsigned> m_returnTypeLocation;
};

} // namespace AST
} // namespace WGSL

SPECIALIZE_TYPE_TRAITS_WGSL_AST(Function)
