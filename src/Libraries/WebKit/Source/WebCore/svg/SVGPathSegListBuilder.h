/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 19, 2023.
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

#include "FloatPoint.h"
#include "SVGPathConsumer.h"

namespace WebCore {

class SVGPathSegList;

class SVGPathSegListBuilder final : public SVGPathConsumer {
public:
    SVGPathSegListBuilder(SVGPathSegList&);

private:
    void incrementPathSegmentCount() final { }
    bool continueConsuming() final { return true; }

    // Used in UnalteredParsing/NormalizedParsing modes.
    void moveTo(const FloatPoint&, bool closed, PathCoordinateMode) final;
    void lineTo(const FloatPoint&, PathCoordinateMode) final;
    void curveToCubic(const FloatPoint&, const FloatPoint&, const FloatPoint&, PathCoordinateMode) final;
    void closePath() final;

    // Only used in UnalteredParsing mode.
    void lineToHorizontal(float, PathCoordinateMode) final;
    void lineToVertical(float, PathCoordinateMode) final;
    void curveToCubicSmooth(const FloatPoint&, const FloatPoint&, PathCoordinateMode) final;
    void curveToQuadratic(const FloatPoint&, const FloatPoint&, PathCoordinateMode) final;
    void curveToQuadraticSmooth(const FloatPoint&, PathCoordinateMode) final;
    void arcTo(float, float, float, bool largeArcFlag, bool sweepFlag, const FloatPoint&, PathCoordinateMode) final;

    SingleThreadWeakRef<SVGPathSegList> m_pathSegList;
};

} // namespace WebCore
