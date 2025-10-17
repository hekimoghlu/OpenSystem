/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 17, 2023.
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

namespace AST {

class StructureMember final : public Node {
    WGSL_AST_BUILDER_NODE(StructureMember);
    friend AttributeValidator;

public:
    using List = ReferenceWrapperVector<StructureMember>;

    NodeKind kind() const final;
    Identifier& name() { return m_name; }
    Identifier& originalName() { return m_originalName; }
    Expression& type() { return m_type; }
    Attribute::List& attributes() { return m_attributes; }

    bool invariant() const { return m_invariant; }
    std::optional<Builtin> builtin() const { return m_builtin; }
    std::optional<unsigned> location() const { return m_location; }
    std::optional<Interpolation> interpolation() const { return m_interpolation; }

    unsigned offset() const { return m_offset; }
    unsigned padding() const { return m_padding; }

    unsigned alignment() const { return *m_alignment; }
    unsigned size() const { return *m_size; }

private:
    StructureMember(SourceSpan span, Identifier&& name, Expression::Ref&& type, Attribute::List&& attributes)
        : Node(span)
        , m_name(WTFMove(name))
        , m_originalName(m_name)
        , m_attributes(WTFMove(attributes))
        , m_type(WTFMove(type))
    { }

    Identifier m_name;
    Identifier m_originalName;
    Attribute::List m_attributes;
    Expression::Ref m_type;

    // Compute
    unsigned m_offset { 0 };
    unsigned m_padding { 0 };

    // Attributes
    bool m_invariant { false };
    std::optional<unsigned> m_alignment;
    std::optional<unsigned> m_size;
    std::optional<Builtin> m_builtin;
    std::optional<unsigned> m_location;
    std::optional<Interpolation> m_interpolation;
};

} // namespace AST
} // namespace WGSL

SPECIALIZE_TYPE_TRAITS_WGSL_AST(StructureMember)
