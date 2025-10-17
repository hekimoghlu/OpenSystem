/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 12, 2025.
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

enum FloatBlendMode {
    BlendHorizontal,
    BlendVertical
};

class SVGPathSource;

class SVGPathBlender {
    WTF_MAKE_TZONE_ALLOCATED(SVGPathBlender);
    WTF_MAKE_NONCOPYABLE(SVGPathBlender);
public:

    static bool addAnimatedPath(SVGPathSource& from, SVGPathSource& to, SVGPathConsumer&, unsigned repeatCount);
    static bool blendAnimatedPath(SVGPathSource& from, SVGPathSource& to, SVGPathConsumer&, float);

    static bool canBlendPaths(SVGPathSource& from, SVGPathSource& to);

private:
    SVGPathBlender(SVGPathSource&, SVGPathSource&, SVGPathConsumer* = nullptr);

    bool canBlendPaths();

    bool addAnimatedPath(unsigned repeatCount);
    bool blendAnimatedPath(float progress);

    bool blendMoveToSegment(float progress);
    bool blendLineToSegment(float progress);
    bool blendLineToHorizontalSegment(float progress);
    bool blendLineToVerticalSegment(float progress);
    bool blendCurveToCubicSegment(float progress);
    bool blendCurveToCubicSmoothSegment(float progress);
    bool blendCurveToQuadraticSegment(float progress);
    bool blendCurveToQuadraticSmoothSegment(float progress);
    bool blendArcToSegment(float progress);

    float blendAnimatedDimensonalFloat(float from, float to, FloatBlendMode, float progress);
    FloatPoint blendAnimatedFloatPoint(const FloatPoint& from, const FloatPoint& to, float progress);

    SingleThreadWeakRef<SVGPathSource> m_fromSource;
    SingleThreadWeakRef<SVGPathSource> m_toSource;
    SingleThreadWeakPtr<SVGPathConsumer> m_consumer; // A null consumer indicates that we're just checking blendability.

    FloatPoint m_fromCurrentPoint;
    FloatPoint m_toCurrentPoint;

    FloatPoint m_fromSubpathPoint;
    FloatPoint m_toSubpathPoint;

    PathCoordinateMode m_fromMode { AbsoluteCoordinates };
    PathCoordinateMode m_toMode { AbsoluteCoordinates };
    unsigned m_addTypesCount { 0 };
    bool m_isInFirstHalfOfAnimation { false };
};

} // namespace WebCore
