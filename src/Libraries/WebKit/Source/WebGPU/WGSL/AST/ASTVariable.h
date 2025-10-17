/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 25, 2025.
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
#include "ASTAttribute.h"
#include "ASTDeclaration.h"
#include "ASTExpression.h"
#include "ASTIdentifier.h"
#include "ASTVariableQualifier.h"

namespace WGSL {
class AttributeValidator;
class RewriteGlobalVariables;
class TypeChecker;
struct Type;

namespace AST {

enum class VariableFlavor : uint8_t {
    Const,
    Let,
    Override,
    Var,
};

enum class VariableRole : uint8_t {
    UserDefined,
    PackedResource,
};

class Variable final : public Declaration {
    WGSL_AST_BUILDER_NODE(Variable);
    friend AttributeValidator;
    friend RewriteGlobalVariables;
    friend TypeChecker;

public:
    using Ref = std::reference_wrapper<Variable>;
    using List = ReferenceWrapperVector<Variable>;

    NodeKind kind() const override;
    VariableFlavor flavor() const { return m_flavor; };
    VariableFlavor& flavor() { return m_flavor; };

    VariableRole role() const { return m_role; }
    VariableRole& role() { return m_role; }

    Identifier& name() override { return m_name; }
    Identifier& originalName() { return m_originalName; }
    Attribute::List& attributes() { return m_attributes; }
    VariableQualifier* maybeQualifier() { return m_qualifier; }
    Expression* maybeTypeName() { return m_type; }
    Expression* maybeInitializer() { return m_initializer; }
    Expression* maybeReferenceType() { return m_referenceType; }
    const Type* storeType() const
    {
        return m_storeType;
    }

    std::optional<AddressSpace> addressSpace() const { return m_addressSpace; }
    std::optional<AccessMode> accessMode() const { return m_accessMode; }
    std::optional<unsigned> binding() const { return m_binding; }
    std::optional<unsigned> group() const { return m_group; }
    std::optional<unsigned> id() const { return m_id; }

private:
    Variable(SourceSpan span, VariableFlavor flavor, Identifier&& name, Expression::Ptr type, Expression::Ptr initializer)
        : Variable(span, flavor, WTFMove(name), { }, type, initializer, { })
    { }

    Variable(SourceSpan span, VariableFlavor flavor, Identifier&& name, VariableQualifier::Ptr qualifier, Expression::Ptr type, Expression::Ptr initializer, Attribute::List&& attributes, VariableRole role = VariableRole::UserDefined)
        : Declaration(span)
        , m_name(WTFMove(name))
        , m_originalName(m_name)
        , m_attributes(WTFMove(attributes))
        , m_qualifier(qualifier)
        , m_type(type)
        , m_initializer(initializer)
        , m_flavor(flavor)
        , m_role(role)
    {
        ASSERT(m_type || m_initializer);
        if (m_type)
            m_storeType = m_type->inferredType();
        else
            m_storeType = m_initializer->inferredType();
    }

    Identifier m_name;
    Identifier m_originalName;
    Attribute::List m_attributes;
    // Each of the following may be null
    // But at least one of type and initializer must be non-null
    VariableQualifier::Ptr m_qualifier;
    Expression::Ptr m_type;
    Expression::Ptr m_initializer;
    VariableFlavor m_flavor;
    VariableRole m_role;
    Expression::Ptr m_referenceType { nullptr };

    // Computed properties
    const Type* m_storeType { nullptr };
    std::optional<AddressSpace> m_addressSpace;
    std::optional<AccessMode> m_accessMode;

    // Attributes
    std::optional<unsigned> m_binding;
    std::optional<unsigned> m_group;
    std::optional<unsigned> m_id;
};

} // namespace AST
} // namespace WGSL

SPECIALIZE_TYPE_TRAITS_WGSL_AST(Variable)
