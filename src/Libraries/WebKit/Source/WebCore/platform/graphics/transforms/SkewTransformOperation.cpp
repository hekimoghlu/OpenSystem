/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 29, 2023.
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
#include "SkewTransformOperation.h"

#include "AnimationUtilities.h"
#include <wtf/text/TextStream.h>

namespace WebCore {

Ref<SkewTransformOperation> SkewTransformOperation::create(double angleX, double angleY, TransformOperation::Type type)
{
    return adoptRef(*new SkewTransformOperation(angleX, angleY, type));
}

SkewTransformOperation::SkewTransformOperation(double angleX, double angleY, TransformOperation::Type type)
    : TransformOperation(type)
    , m_angleX(angleX)
    , m_angleY(angleY)
{
    RELEASE_ASSERT(isSkewTransformOperationType(type));
}

bool SkewTransformOperation::operator==(const TransformOperation& other) const
{
    if (!isSameType(other))
        return false;
    const SkewTransformOperation& s = downcast<SkewTransformOperation>(other);
    return m_angleX == s.m_angleX && m_angleY == s.m_angleY;
}

Ref<TransformOperation> SkewTransformOperation::blend(const TransformOperation* from, const BlendingContext& context, bool blendToIdentity)
{
    if (blendToIdentity)
        return SkewTransformOperation::create(WebCore::blend(m_angleX, 0.0, context), WebCore::blend(m_angleY, 0.0, context), type());

    auto outputType = sharedPrimitiveType(from);
    if (!outputType)
        return *this;

    const SkewTransformOperation* fromOp = downcast<SkewTransformOperation>(from);
    double fromAngleX = fromOp ? fromOp->m_angleX : 0;
    double fromAngleY = fromOp ? fromOp->m_angleY : 0;
    return SkewTransformOperation::create(WebCore::blend(fromAngleX, m_angleX, context), WebCore::blend(fromAngleY, m_angleY, context), *outputType);
}

void SkewTransformOperation::dump(TextStream& ts) const
{
    ts << type() << "(" << TextStream::FormatNumberRespectingIntegers(m_angleX) << "deg, " << TextStream::FormatNumberRespectingIntegers(m_angleY) << "deg)";
}

} // namespace WebCore
