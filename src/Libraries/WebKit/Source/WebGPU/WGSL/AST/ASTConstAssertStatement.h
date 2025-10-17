/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 23, 2024.
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

#include "ASTBuilder.h"
#include "ASTConstAssert.h"
#include "ASTStatement.h"

namespace WGSL::AST {

class ConstAssertStatement final : public Statement {
    WGSL_AST_BUILDER_NODE(ConstAssertStatement);

public:
    NodeKind kind() const override;
    ConstAssert& assertion() { return m_assertion.get(); }

private:
    ConstAssertStatement(SourceSpan span, AST::ConstAssert::Ref&& assertion)
        : Statement(span)
        , m_assertion(WTFMove(assertion))
    { }

    AST::ConstAssert::Ref m_assertion;
};

} // namespace WGSL::AST

SPECIALIZE_TYPE_TRAITS_WGSL_AST(ConstAssertStatement)
