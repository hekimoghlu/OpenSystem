/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 21, 2025.
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

#include "ASTStatement.h"

namespace WGSL::AST {

struct SwitchClause {
    AST::Expression::List selectors;
    AST::CompoundStatement::Ref body;
};

class SwitchStatement final : public Statement {
    WGSL_AST_BUILDER_NODE(SwitchStatement);
public:
    NodeKind kind() const final;
    Expression& value() { return m_value.get(); }
    Attribute::List& valueAttributes() { return m_valueAttributes; }
    Vector<SwitchClause>& clauses() { return m_clauses; }
    SwitchClause& defaultClause() { return m_defaultClause; }

    bool isInsideLoop() const { return m_isInsideLoop; }
    void setIsInsideLoop() { m_isInsideLoop = true;; }

    bool isNestedInsideLoop() const { return m_isNestedInsideLoop; }
    void setIsNestedInsideLoop() { m_isNestedInsideLoop = true; }

private:
    SwitchStatement(SourceSpan span, AST::Expression::Ref&& value, AST::Attribute::List&& valueAttributes, Vector<SwitchClause>&& clauses, SwitchClause&& defaultClause)
        : Statement(span)
        , m_value(WTFMove(value))
        , m_valueAttributes(WTFMove(valueAttributes))
        , m_clauses(WTFMove(clauses))
        , m_defaultClause(WTFMove(defaultClause))
    { }

    bool m_isInsideLoop { false };
    bool m_isNestedInsideLoop { false };
    Expression::Ref m_value;
    Attribute::List m_valueAttributes;
    Vector<SwitchClause> m_clauses;
    SwitchClause m_defaultClause;
};

} // namespace WGSL::AST

SPECIALIZE_TYPE_TRAITS_WGSL_AST(SwitchStatement)
