/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 15, 2022.
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
#include "AcceleratedEffectValues.h"

#if ENABLE(THREADED_ANIMATION_RESOLUTION)

#include "IntSize.h"
#include "LengthFunctions.h"
#include "MotionPath.h"
#include "Path.h"
#include "RenderElementInlines.h"
#include "RenderLayerModelObject.h"
#include "RenderStyleInlines.h"
#include "TransformOperationData.h"

namespace WebCore {

AcceleratedEffectValues AcceleratedEffectValues::clone() const
{
    std::optional<TransformOperationData> clonedTransformOperationData;
    if (transformOperationData)
        clonedTransformOperationData = transformOperationData;

    auto clonedTransformOrigin = transformOrigin;
    auto clonedTransform = transform.clone();

    RefPtr<TransformOperation> clonedTranslate;
    if (auto* srcTranslate = translate.get())
        clonedTranslate = srcTranslate->clone();

    RefPtr<TransformOperation> clonedScale;
    if (auto* srcScale = scale.get())
        clonedScale = srcScale->clone();

    RefPtr<TransformOperation> clonedRotate;
    if (auto* srcRotate = rotate.get())
        clonedRotate = srcRotate->clone();

    RefPtr<PathOperation> clonedOffsetPath;
    if (auto* srcOffsetPath = offsetPath.get())
        clonedOffsetPath = srcOffsetPath->clone();

    auto clonedOffsetDistance = offsetDistance;
    auto clonedOffsetPosition = offsetPosition;
    auto clonedOffsetAnchor = offsetAnchor;
    auto clonedOffsetRotate = offsetRotate;

    auto clonedFilter = filter.clone();
    auto clonedBackdropFilter = backdropFilter.clone();

    return {
        opacity,
        WTFMove(clonedTransformOperationData),
        WTFMove(clonedTransformOrigin),
        transformBox,
        WTFMove(clonedTransform),
        WTFMove(clonedTranslate),
        WTFMove(clonedScale),
        WTFMove(clonedRotate),
        WTFMove(clonedOffsetPath),
        WTFMove(clonedOffsetDistance),
        WTFMove(clonedOffsetPosition),
        WTFMove(clonedOffsetAnchor),
        WTFMove(clonedOffsetRotate),
        WTFMove(clonedFilter),
        WTFMove(clonedBackdropFilter)
    };
}

static LengthPoint nonCalculatedLengthPoint(LengthPoint lengthPoint, const IntSize& borderBoxSize)
{
    if (!lengthPoint.x.isCalculated() && !lengthPoint.y.isCalculated())
        return lengthPoint;
    return {
        { floatValueForLength(lengthPoint.x, borderBoxSize.width()), LengthType::Fixed },
        { floatValueForLength(lengthPoint.y, borderBoxSize.height()), LengthType::Fixed }
    };
}

AcceleratedEffectValues::AcceleratedEffectValues(const RenderStyle& style, const IntRect& borderBoxRect, const RenderLayerModelObject* renderer)
{
    opacity = style.opacity();

    auto borderBoxSize = borderBoxRect.size();

    if (renderer)
        transformOperationData = TransformOperationData(renderer->transformReferenceBoxRect(style), renderer);

    transformBox = style.transformBox();
    transform = style.transform().selfOrCopyWithResolvedCalculatedValues(borderBoxSize);

    if (auto* srcTranslate = style.translate())
        translate = srcTranslate->selfOrCopyWithResolvedCalculatedValues(borderBoxSize);
    if (auto* srcScale = style.scale())
        scale = srcScale->selfOrCopyWithResolvedCalculatedValues(borderBoxSize);
    if (auto* srcRotate = style.rotate())
        rotate = srcRotate->selfOrCopyWithResolvedCalculatedValues(borderBoxSize);
    transformOrigin = nonCalculatedLengthPoint(style.transformOriginXY(), borderBoxSize);

    offsetPath = style.offsetPath();
    offsetPosition = nonCalculatedLengthPoint(style.offsetPosition(), borderBoxSize);
    offsetAnchor = nonCalculatedLengthPoint(style.offsetAnchor(), borderBoxSize);
    offsetRotate = style.offsetRotate();
    offsetDistance = style.offsetDistance();
    if (offsetDistance.isCalculated() && offsetPath) {
        auto anchor = borderBoxRect.location() + floatPointForLengthPoint(transformOrigin, borderBoxSize);
        if (!offsetAnchor.x.isAuto())
            anchor = floatPointForLengthPoint(offsetAnchor, borderBoxRect.size()) + borderBoxRect.location();

        auto path = offsetPath->getPath(TransformOperationData(FloatRect(borderBoxRect)));
        offsetDistance = { path ? path->length() : 0.0f, LengthType:: Fixed };
    }

    filter = style.filter();
    backdropFilter = style.backdropFilter();
}

TransformationMatrix AcceleratedEffectValues::computedTransformationMatrix(const FloatRect& boundingBox) const
{
    // https://www.w3.org/TR/css-transforms-2/#ctm
    // The transformation matrix is computed from the transform, transform-origin, translate, rotate, scale, and offset properties as follows:
    // 1. Start with the identity matrix.
    TransformationMatrix matrix;

    // 2. Translate by the computed X, Y, and Z values of transform-origin.
    // (not needed, the GraphicsLayer handles that)

    // 3. Translate by the computed X, Y, and Z values of translate.
    if (translate)
        translate->apply(matrix, boundingBox.size());

    // 4. Rotate by the computed <angle> about the specified axis of rotate.
    if (rotate)
        rotate->apply(matrix, boundingBox.size());

    // 5. Scale by the computed X, Y, and Z values of scale.
    if (scale)
        scale->apply(matrix, boundingBox.size());

    // 6. Translate and rotate by the transform specified by offset.
    if (transformOperationData && offsetPath) {
        auto computedTransformOrigin = boundingBox.location() + floatPointForLengthPoint(transformOrigin, boundingBox.size());
        MotionPath::applyMotionPathTransform(matrix, *transformOperationData, computedTransformOrigin, *offsetPath, offsetAnchor, offsetDistance, offsetRotate, transformBox);
    }

    // 7. Multiply by each of the transform functions in transform from left to right.
    transform.apply(matrix, boundingBox.size());

    // 8. Translate by the negated computed X, Y and Z values of transform-origin.
    // (not needed, the GraphicsLayer handles that)

    return matrix;
}

} // namespace WebCore

#endif // ENABLE(THREADED_ANIMATION_RESOLUTION)
