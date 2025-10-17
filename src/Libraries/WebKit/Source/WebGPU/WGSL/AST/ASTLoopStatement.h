/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 7, 2021.
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

namespace WGSL::AST {

struct Continuing {
    Statement::List body;
    Attribute::List attributes;
    Expression::Ptr breakIf;
};

class LoopStatement final : public Statement {
    WGSL_AST_BUILDER_NODE(LoopStatement);
public:
    NodeKind kind() const override;
    Attribute::List& attributes() { return m_attributes; }
    Statement::List& body() { return m_body; }
    std::optional<Continuing>& continuing() { return m_continuing; }

    void setContainsSwitch() { m_containsSwitch = true; }
    bool containsSwitch() const { return m_containsSwitch; }

private:
    LoopStatement(SourceSpan span, Attribute::List&& attributes, Statement::List&& body, std::optional<Continuing>&& continuing)
        : Statement(span)
        , m_attributes(WTFMove(attributes))
        , m_body(WTFMove(body))
        , m_continuing(WTFMove(continuing))
    { }

    Attribute::List m_attributes;
    Statement::List m_body;
    std::optional<Continuing> m_continuing;

    bool m_containsSwitch { false };
};

} // namespace WGSL::AST

SPECIALIZE_TYPE_TRAITS_WGSL_AST(LoopStatement)
