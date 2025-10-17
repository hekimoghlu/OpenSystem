/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 26, 2024.
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

#if ENABLE(THREADED_ANIMATION_RESOLUTION)

#include "FilterOperations.h"
#include "Length.h"
#include "LengthPoint.h"
#include "OffsetRotation.h"
#include "PathOperation.h"
#include "RenderStyleConstants.h"
#include "RotateTransformOperation.h"
#include "ScaleTransformOperation.h"
#include "TransformOperations.h"
#include "TransformationMatrix.h"
#include "TranslateTransformOperation.h"

namespace WebCore {

class IntRect;
class Path;
class RenderLayerModelObject;
class RenderStyle;

struct AcceleratedEffectValues {
    float opacity { 1 };
    std::optional<TransformOperationData> transformOperationData;
    LengthPoint transformOrigin { };
    TransformBox transformBox { TransformBox::ContentBox };
    TransformOperations transform { };
    RefPtr<TransformOperation> translate;
    RefPtr<TransformOperation> scale;
    RefPtr<TransformOperation> rotate;
    RefPtr<PathOperation> offsetPath;
    Length offsetDistance { };
    LengthPoint offsetPosition { };
    LengthPoint offsetAnchor { };
    OffsetRotation offsetRotate { };
    FilterOperations filter { };
    FilterOperations backdropFilter { };

    AcceleratedEffectValues() = default;
    AcceleratedEffectValues(const RenderStyle&, const IntRect&, const RenderLayerModelObject* = nullptr);
    AcceleratedEffectValues(float opacity, std::optional<TransformOperationData>&& transformOperationData, LengthPoint&& transformOrigin, TransformBox transformBox, TransformOperations&& transform, RefPtr<TransformOperation>&& translate, RefPtr<TransformOperation>&& scale, RefPtr<TransformOperation>&& rotate, RefPtr<PathOperation>&& offsetPath, Length&& offsetDistance, LengthPoint&& offsetPosition, LengthPoint&& offsetAnchor, OffsetRotation&& offsetRotate, FilterOperations&& filter, FilterOperations&& backdropFilter)
        : opacity(opacity)
        , transformOperationData(WTFMove(transformOperationData))
        , transformOrigin(WTFMove(transformOrigin))
        , transformBox(transformBox)
        , transform(WTFMove(transform))
        , translate(WTFMove(translate))
        , scale(WTFMove(scale))
        , rotate(WTFMove(rotate))
        , offsetPath(WTFMove(offsetPath))
        , offsetDistance(WTFMove(offsetDistance))
        , offsetPosition(WTFMove(offsetPosition))
        , offsetAnchor(WTFMove(offsetAnchor))
        , offsetRotate(WTFMove(offsetRotate))
        , filter(WTFMove(filter))
        , backdropFilter(WTFMove(backdropFilter))
    {
    }

    WEBCORE_EXPORT AcceleratedEffectValues clone() const;
    WEBCORE_EXPORT TransformationMatrix computedTransformationMatrix(const FloatRect&) const;
};

} // namespace WebCore

#endif // ENABLE(THREADED_ANIMATION_RESOLUTION)
