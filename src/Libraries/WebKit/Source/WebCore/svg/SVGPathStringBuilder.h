/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 28, 2024.
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
#include <wtf/text/StringBuilder.h>

namespace WebCore {

class SVGPathStringBuilder final : public SVGPathConsumer {
public:
    WEBCORE_EXPORT SVGPathStringBuilder();
    WEBCORE_EXPORT virtual ~SVGPathStringBuilder();

    WEBCORE_EXPORT String result();

    void incrementPathSegmentCount() final;
    bool continueConsuming() final;

    // Used in UnalteredParsing/NormalizedParsing modes.
    WEBCORE_EXPORT void moveTo(const FloatPoint&, bool closed, PathCoordinateMode) final;
    WEBCORE_EXPORT void lineTo(const FloatPoint&, PathCoordinateMode) final;
    WEBCORE_EXPORT void curveToCubic(const FloatPoint&, const FloatPoint&, const FloatPoint&, PathCoordinateMode) final;
    WEBCORE_EXPORT void closePath() final;

    // Only used in UnalteredParsing mode.
    void lineToHorizontal(float, PathCoordinateMode) final;
    void lineToVertical(float, PathCoordinateMode) final;
    void curveToCubicSmooth(const FloatPoint&, const FloatPoint&, PathCoordinateMode) final;
    WEBCORE_EXPORT void curveToQuadratic(const FloatPoint&, const FloatPoint&, PathCoordinateMode) final;
    void curveToQuadraticSmooth(const FloatPoint&, PathCoordinateMode) final;
    void arcTo(float, float, float, bool largeArcFlag, bool sweepFlag, const FloatPoint&, PathCoordinateMode) final;

private:
    StringBuilder m_stringBuilder;
};

} // namespace WebCore
