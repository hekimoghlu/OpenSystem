/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 21, 2023.
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

namespace WGSL::AST {

class IdAttribute final : public Attribute {
    WGSL_AST_BUILDER_NODE(IdAttribute);

public:
    NodeKind kind() const override;
    Expression& value() { return m_value; }

private:
    IdAttribute(SourceSpan span, Expression::Ref&& value)
        : Attribute(span)
        , m_value(WTFMove(value))
    { }

    Expression::Ref m_value;
};

} // namespace WGSL::AST

SPECIALIZE_TYPE_TRAITS_WGSL_AST(IdAttribute)
