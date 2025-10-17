/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 19, 2024.
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

#include "SVGPathConsumer.h"

namespace WebCore {

class FloatPoint;
class PathTraversalState;

class SVGPathTraversalStateBuilder final : public SVGPathConsumer {
public:
    SVGPathTraversalStateBuilder(PathTraversalState&, float desiredLength = 0);

    unsigned pathSegmentIndex() const { return m_segmentIndex; }
    float totalLength() const;
    FloatPoint currentPoint() const;

    void incrementPathSegmentCount() final { ++m_segmentIndex; }
    bool continueConsuming() final;

private:
    // Used in UnalteredParsing/NormalizedParsing modes.
    void moveTo(const FloatPoint&, bool closed, PathCoordinateMode) final;
    void lineTo(const FloatPoint&, PathCoordinateMode) final;
    void curveToCubic(const FloatPoint&, const FloatPoint&, const FloatPoint&, PathCoordinateMode) final;
    void closePath() final;

private:
    // Not used for PathTraversalState.
    void lineToHorizontal(float, PathCoordinateMode) final { ASSERT_NOT_REACHED(); }
    void lineToVertical(float, PathCoordinateMode) final { ASSERT_NOT_REACHED(); }
    void curveToCubicSmooth(const FloatPoint&, const FloatPoint&, PathCoordinateMode) final { ASSERT_NOT_REACHED(); }
    void curveToQuadratic(const FloatPoint&, const FloatPoint&, PathCoordinateMode) final { ASSERT_NOT_REACHED(); }
    void curveToQuadraticSmooth(const FloatPoint&, PathCoordinateMode) final { ASSERT_NOT_REACHED(); }
    void arcTo(float, float, float, bool, bool, const FloatPoint&, PathCoordinateMode) final { ASSERT_NOT_REACHED(); }

    PathTraversalState& m_traversalState;
    unsigned m_segmentIndex { 0 };
};

} // namespace WebCore
