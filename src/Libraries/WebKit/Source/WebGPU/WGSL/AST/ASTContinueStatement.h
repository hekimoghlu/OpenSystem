/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 6, 2025.
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

#pragma once

#include "ASTStatement.h"

namespace WGSL::AST {

class ContinueStatement final : public Statement {
    WGSL_AST_BUILDER_NODE(ContinueStatement);
public:
    NodeKind kind() const override;

    void setIsFromSwitchToContinuing() { m_isFromSwitchToContinuing = true; };
    bool isFromSwitchToContinuing() const { return m_isFromSwitchToContinuing; };

private:
    ContinueStatement(SourceSpan span)
        : Statement(span)
    { }

    bool m_isFromSwitchToContinuing { false };
};

} // namespace WGSL::AST

SPECIALIZE_TYPE_TRAITS_WGSL_AST(ContinueStatement)
