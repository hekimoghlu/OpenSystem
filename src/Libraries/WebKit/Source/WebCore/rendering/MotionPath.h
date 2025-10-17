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
#pragma once

#include "FloatRoundedRect.h"
#include "RenderLayerModelObject.h"

namespace WebCore {

class BoxPathOperation;
class RenderElement;
class PathOperation;
class RayPathOperation;
class ShapePathOperation;

struct TransformOperationData;

struct MotionPathData {
    FloatRoundedRect containingBlockBoundingRect;
    FloatPoint offsetFromContainingBlock;
    FloatPoint usedStartingPosition;
};

class MotionPath {
public:
    static std::optional<MotionPathData> motionPathDataForRenderer(const RenderElement&);
    static bool needsUpdateAfterContainingBlockLayout(const PathOperation&);

    static void applyMotionPathTransform(TransformationMatrix&, const TransformOperationData&, const FloatPoint& transformOrigin, const PathOperation&, const LengthPoint& offsetAnchor, const Length& offsetDistance, const OffsetRotation&, TransformBox);
    static void applyMotionPathTransform(const RenderStyle&, const TransformOperationData&, TransformationMatrix&);

    static std::optional<Path> computePathForBox(const BoxPathOperation&, const TransformOperationData&);
    static std::optional<Path> computePathForShape(const ShapePathOperation&, const TransformOperationData&);
    static std::optional<Path> computePathForRay(const RayPathOperation&, const TransformOperationData&);
};

} // namespace WebCore
