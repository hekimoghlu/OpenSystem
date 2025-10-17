/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 12, 2023.
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

namespace WGSL::AST {

struct WorkgroupSize {
    Expression::Ptr x;
    Expression::Ptr y;
    Expression::Ptr z;
};

class WorkgroupSizeAttribute final : public Attribute {
    WGSL_AST_BUILDER_NODE(WorkgroupSizeAttribute);

public:
    NodeKind kind() const override;

    Expression& x() { return *m_workgroupSize.x; }
    Expression* maybeY() { return m_workgroupSize.y; }
    Expression* maybeZ() { return m_workgroupSize.z; }

    const Expression& x() const { return *m_workgroupSize.x; }
    const Expression* maybeY() const { return m_workgroupSize.y; }
    const Expression* maybeZ() const { return m_workgroupSize.z; }

    const WorkgroupSize& workgroupSize() const { return m_workgroupSize; }

private:
    WorkgroupSizeAttribute(SourceSpan span, Expression::Ref&& x, Expression::Ptr maybeY, Expression::Ptr maybeZ)
        : Attribute(span)
        , m_workgroupSize({ &x.get(), maybeY, maybeZ })
    { }

    WorkgroupSize m_workgroupSize;
};

} // namespace WGSL::AST

SPECIALIZE_TYPE_TRAITS_WGSL_AST(WorkgroupSizeAttribute)
