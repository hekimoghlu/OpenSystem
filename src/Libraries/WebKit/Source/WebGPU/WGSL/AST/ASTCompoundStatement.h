/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 4, 2023.
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

class CompoundStatement final : public Statement {
    WGSL_AST_BUILDER_NODE(CompoundStatement);
public:
    using Ref = std::reference_wrapper<CompoundStatement>;

    NodeKind kind() const override;
    Attribute::List& attributes() { return m_attributes; }
    Statement::List& statements() { return m_statements; }
    const Statement::List& statements() const { return m_statements; }

private:
    CompoundStatement(SourceSpan span, Attribute::List&& attributes, Statement::List&& statements)
        : Statement(span)
        , m_attributes(WTFMove(attributes))
        , m_statements(WTFMove(statements))
    { }

    Attribute::List m_attributes;
    Statement::List m_statements;
};

} // namespace WGSL::AST

SPECIALIZE_TYPE_TRAITS_WGSL_AST(CompoundStatement)
