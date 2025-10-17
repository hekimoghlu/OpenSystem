/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 29, 2021.
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
#include "ScaleTransformOperation.h"

#include "AnimationUtilities.h"
#include <wtf/text/TextStream.h>

namespace WebCore {

Ref<ScaleTransformOperation> ScaleTransformOperation::create(double sx, double sy, double sz, TransformOperation::Type type)
{
    return adoptRef(*new ScaleTransformOperation(sx, sy, sz, type));
}

ScaleTransformOperation::ScaleTransformOperation(double sx, double sy, double sz, TransformOperation::Type type)
    : TransformOperation(type)
    , m_x(sx)
    , m_y(sy)
    , m_z(sz)
{
    RELEASE_ASSERT(isScaleTransformOperationType(type));
}

bool ScaleTransformOperation::operator==(const TransformOperation& other) const
{
    if (!isSameType(other))
        return false;
    const ScaleTransformOperation& s = downcast<ScaleTransformOperation>(other);
    return m_x == s.m_x && m_y == s.m_y && m_z == s.m_z;
}

static double blendScaleComponent(double from, double to, const BlendingContext& context)
{
    switch (context.compositeOperation) {
    case CompositeOperation::Replace:
        return WebCore::blend(from, to, context);
    case CompositeOperation::Add:
        ASSERT(context.progress == 1.0);
        return from * to;
    case CompositeOperation::Accumulate:
        ASSERT(context.progress == 1.0);
        return from + to - 1;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

Ref<TransformOperation> ScaleTransformOperation::blend(const TransformOperation* from, const BlendingContext& context, bool blendToIdentity)
{
    if (blendToIdentity)
        return ScaleTransformOperation::create(blendScaleComponent(m_x, 1.0, context), blendScaleComponent(m_y, 1.0, context), blendScaleComponent(m_z, 1.0, context), type());

    auto outputType = sharedPrimitiveType(from);
    if (!outputType)
        return *this;

    const ScaleTransformOperation* fromOp = downcast<ScaleTransformOperation>(from);
    double fromX = fromOp ? fromOp->m_x : 1.0;
    double fromY = fromOp ? fromOp->m_y : 1.0;
    double fromZ = fromOp ? fromOp->m_z : 1.0;
    return ScaleTransformOperation::create(blendScaleComponent(fromX, m_x, context), blendScaleComponent(fromY, m_y, context), blendScaleComponent(fromZ, m_z, context), *outputType);
}

void ScaleTransformOperation::dump(TextStream& ts) const
{
    ts << type() << "(" << TextStream::FormatNumberRespectingIntegers(m_x) << ", " << TextStream::FormatNumberRespectingIntegers(m_y) << ", " << TextStream::FormatNumberRespectingIntegers(m_z) << ")";
}

} // namespace WebCore
