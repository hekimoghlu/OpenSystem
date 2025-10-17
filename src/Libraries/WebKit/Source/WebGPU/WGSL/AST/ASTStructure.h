/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 28, 2024.
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
#include "ASTDeclaration.h"
#include "ASTIdentifier.h"
#include "ASTStructureMember.h"

namespace WGSL {

class AttributeValidator;
class RewriteGlobalVariables;
class TypeChecker;

namespace AST {

enum class StructureRole : uint8_t {
    UserDefined,
    VertexInput,
    FragmentInput,
    ComputeInput,
    VertexOutput,
    BindGroup,
    UserDefinedResource,
    PackedResource,
    FragmentOutput,
    FragmentOutputWrapper,
};

class Structure final : public Declaration {
    WGSL_AST_BUILDER_NODE(Structure);
    friend AttributeValidator;
    friend RewriteGlobalVariables;
    friend TypeChecker;

public:
    using Ref = std::reference_wrapper<Structure>;
    using List = ReferenceWrapperVector<Structure>;

    NodeKind kind() const override;
    StructureRole role() const { return m_role; }
    StructureRole& role() { return m_role; }
    Identifier& name() override { return m_name; }
    Attribute::List& attributes() { return m_attributes; }
    StructureMember::List& members() { return m_members; }
    Structure* original() const { return m_original; }
    Structure* packed() const { return m_packed; }
    const Type* inferredType() const { return m_inferredType; }

    void setRole(StructureRole role) { m_role = role; }

    bool hasSizeOrAlignmentAttributes() const { return m_hasSizeOrAlignmentAttributes; }
    unsigned size() const { return *m_size; }
    unsigned alignment() const { return *m_alignment; }

private:
    Structure(SourceSpan span, Identifier&& name, StructureMember::List&& members, Attribute::List&& attributes, StructureRole role, Structure* original = nullptr)
        : Declaration(span)
        , m_name(WTFMove(name))
        , m_attributes(WTFMove(attributes))
        , m_members(WTFMove(members))
        , m_role(role)
        , m_original(original)
    {
        if (m_original) {
            ASSERT(m_role == StructureRole::PackedResource);
            m_original->m_packed = this;
            m_size = original->m_size;
            m_alignment = original->m_alignment;
        }
    }

    Identifier m_name;
    Attribute::List m_attributes;
    StructureMember::List m_members;
    StructureRole m_role;
    Structure* m_original;
    Structure* m_packed { nullptr };
    const Type* m_inferredType { nullptr };

    // Computed properties
    bool m_hasSizeOrAlignmentAttributes { false };
    std::optional<unsigned> m_size;
    std::optional<unsigned> m_alignment;
};

} // namespace AST
} // namespace WGSL

SPECIALIZE_TYPE_TRAITS_WGSL_AST(Structure)
