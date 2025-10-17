/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 5, 2024.
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

// This is a special node used when modifying the AST through compiler passes.
// The passes can replace one node with another, but only if the size of the new
// class fits in the existing space. This is a small node used when needing to insert a
// larger node than the one current in the tree. E.g. replacing an identifier
// with a structure access.
class IdentityExpression final : public Expression {
    WGSL_AST_BUILDER_NODE(IdentityExpression);
public:
    NodeKind kind() const override;
    Expression& expression() { return m_expression.get(); }

private:
    IdentityExpression(SourceSpan span, Expression::Ref&& expression)
        : Expression(span)
        , m_expression(WTFMove(expression))
    {
        m_inferredType = m_expression.get().inferredType();
    }

    Expression::Ref m_expression;
};

} // namespace WGSL::AST

SPECIALIZE_TYPE_TRAITS_WGSL_AST(IdentityExpression)
