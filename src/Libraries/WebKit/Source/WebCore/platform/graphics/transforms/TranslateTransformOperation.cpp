/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 23, 2023.
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
#include "TranslateTransformOperation.h"

#include "AnimationUtilities.h"
#include "FloatConversion.h"
#include <wtf/text/TextStream.h>

namespace WebCore {

Ref<TranslateTransformOperation> TranslateTransformOperation::create(const Length& tx, const Length& ty, const Length& tz, TransformOperation::Type type)
{
    return adoptRef(*new TranslateTransformOperation(tx, ty, tz, type));
}

TranslateTransformOperation::TranslateTransformOperation(const Length& tx, const Length& ty, const Length& tz, TransformOperation::Type type)
    : TransformOperation(type)
    , m_x(tx)
    , m_y(ty)
    , m_z(tz)
{
    RELEASE_ASSERT(isTranslateTransformOperationType(type));
}

bool TranslateTransformOperation::operator==(const TransformOperation& other) const
{
    if (!isSameType(other))
        return false;
    const TranslateTransformOperation& t = downcast<TranslateTransformOperation>(other);
    return m_x == t.m_x && m_y == t.m_y && m_z == t.m_z;
}

Ref<TransformOperation> TranslateTransformOperation::blend(const TransformOperation* from, const BlendingContext& context, bool blendToIdentity)
{
    Length zeroLength(0, LengthType::Fixed);
    if (blendToIdentity)
        return TranslateTransformOperation::create(WebCore::blend(m_x, zeroLength, context), WebCore::blend(m_y, zeroLength, context), WebCore::blend(m_z, zeroLength, context), type());

    auto outputType = sharedPrimitiveType(from);
    if (!outputType)
        return *this;

    const TranslateTransformOperation* fromOp = downcast<TranslateTransformOperation>(from);
    Length fromX = fromOp ? fromOp->m_x : zeroLength;
    Length fromY = fromOp ? fromOp->m_y : zeroLength;
    Length fromZ = fromOp ? fromOp->m_z : zeroLength;
    return TranslateTransformOperation::create(WebCore::blend(fromX, x(), context), WebCore::blend(fromY, y(), context), WebCore::blend(fromZ, z(), context), *outputType);
}

void TranslateTransformOperation::dump(TextStream& ts) const
{
    ts << type() << "(" << m_x << ", " << m_y << ", " << m_z << ")";
}

Ref<TransformOperation> TranslateTransformOperation::selfOrCopyWithResolvedCalculatedValues(const FloatSize& borderBoxSize)
{
    if (!m_x.isCalculated() && !m_y.isCalculated() && !m_z.isCalculated())
        return TransformOperation::selfOrCopyWithResolvedCalculatedValues(borderBoxSize);

    Length x = { xAsFloat(borderBoxSize), LengthType::Fixed };
    Length y = { yAsFloat(borderBoxSize), LengthType::Fixed };
    return create(x, y, m_z, type());
}

} // namespace WebCore
