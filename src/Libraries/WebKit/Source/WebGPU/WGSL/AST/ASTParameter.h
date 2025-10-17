/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 25, 2021.
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
#include "ASTBuilder.h"
#include "ASTExpression.h"
#include "ASTIdentifier.h"
#include "ASTInterpolateAttribute.h"
#include <wtf/ReferenceWrapperVector.h>

namespace WGSL {

class AttributeValidator;
class EntryPointRewriter;

namespace AST {

enum class ParameterRole : uint8_t {
    UserDefined,
    StageIn,
    BindGroup,
    PackedResource,
};

class Parameter final : public Node {
    WGSL_AST_BUILDER_NODE(Parameter);
    friend AttributeValidator;
    friend EntryPointRewriter;

public:
    using List = ReferenceWrapperVector<Parameter>;

    NodeKind kind() const override;
    ParameterRole role() const { return m_role; }

    Identifier& name() { return m_name; }
    Expression& typeName() { return m_typeName.get(); }
    Attribute::List& attributes() { return m_attributes; }

    const Identifier& name() const { return m_name; }
    const Expression& typeName() const { return m_typeName.get(); }
    const Attribute::List& attributes() const { return m_attributes; }

    bool invariant() const { return m_invariant; }
    std::optional<Builtin> builtin() const { return m_builtin; }
    std::optional<Interpolation> interpolation() const { return m_interpolation; }
    std::optional<unsigned> location() const { return m_location; }

private:
    Parameter(SourceSpan span, Identifier&& name, Expression::Ref&& typeName, Attribute::List&& attributes, ParameterRole role)
        : Node(span)
        , m_role(role)
        , m_name(WTFMove(name))
        , m_typeName(WTFMove(typeName))
        , m_attributes(WTFMove(attributes))
    { }

    ParameterRole m_role;
    Identifier m_name;
    Expression::Ref m_typeName;
    Attribute::List m_attributes;

    // Attributes
    bool m_invariant { false };
    std::optional<Builtin> m_builtin;
    std::optional<Interpolation> m_interpolation;
    std::optional<unsigned> m_location;
};

} // namespace AST
} // namespace WGSL

SPECIALIZE_TYPE_TRAITS_WGSL_AST(Parameter)
