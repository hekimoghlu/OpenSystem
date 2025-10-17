/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 18, 2022.
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
#include "config.h"
#include "MatrixTransformOperation.h"

#include "AnimationUtilities.h"
#include <algorithm>
#include <wtf/text/TextStream.h>

namespace WebCore {

Ref<MatrixTransformOperation> MatrixTransformOperation::create(const TransformationMatrix& t)
{
    return adoptRef(*new MatrixTransformOperation(t));
}

MatrixTransformOperation::MatrixTransformOperation(const TransformationMatrix& t)
    : TransformOperation(TransformOperation::Type::Matrix)
    , m_a(t.a())
    , m_b(t.b())
    , m_c(t.c())
    , m_d(t.d())
    , m_e(t.e())
    , m_f(t.f())
{
}

bool MatrixTransformOperation::operator==(const TransformOperation& other) const
{
    if (!isSameType(other))
        return false;
    const MatrixTransformOperation& m = downcast<MatrixTransformOperation>(other);
    return m_a == m.m_a && m_b == m.m_b && m_c == m.m_c && m_d == m.m_d && m_e == m.m_e && m_f == m.m_f;
}

Ref<TransformOperation> MatrixTransformOperation::blend(const TransformOperation* from, const BlendingContext& context, bool blendToIdentity)
{
    auto createOperation = [] (TransformationMatrix& to, TransformationMatrix& from, const BlendingContext& context) {
        to.blend(from, context.progress, context.compositeOperation);
        return MatrixTransformOperation::create(to);
    };

    if (!sharedPrimitiveType(from))
        return *this;

    // convert the TransformOperations into matrices
    TransformationMatrix fromT;
    TransformationMatrix toT(m_a, m_b, m_c, m_d, m_e, m_f);
    if (from) {
        const MatrixTransformOperation& m = downcast<MatrixTransformOperation>(*from);
        fromT.setMatrix(m.m_a, m.m_b, m.m_c, m.m_d, m.m_e, m.m_f);
    }

    if (blendToIdentity)
        return createOperation(fromT, toT, context);
    return createOperation(toT, fromT, context);
}

void MatrixTransformOperation::dump(TextStream& ts) const
{
    ts << "("
        << TextStream::FormatNumberRespectingIntegers(m_a) << ", "
        << TextStream::FormatNumberRespectingIntegers(m_b) << ", "
        << TextStream::FormatNumberRespectingIntegers(m_c) << ", "
        << TextStream::FormatNumberRespectingIntegers(m_d) << ", "
        << TextStream::FormatNumberRespectingIntegers(m_e) << ", "
        << TextStream::FormatNumberRespectingIntegers(m_f) << ")";
}

} // namespace WebCore
