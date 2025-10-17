/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 30, 2022.
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
#include "SVGPathSeg.h"
#include <wtf/text/WTFString.h>

namespace WebCore {

class SVGPathByteStream;
class SVGPathSource;

class SVGPathParser {
public:
    static bool parse(SVGPathSource&, SVGPathConsumer&, PathParsingMode = NormalizedParsing, bool checkForInitialMoveTo = true);

    static bool parseToByteStream(SVGPathSource&, SVGPathByteStream&, PathParsingMode = NormalizedParsing, bool checkForInitialMoveTo = true);
    static bool parseToString(SVGPathSource&, String& result, PathParsingMode = NormalizedParsing, bool checkForInitialMoveTo = true);

private:
    SVGPathParser(SVGPathConsumer&, SVGPathSource&, PathParsingMode);
    bool parsePathData(bool checkForInitialMoveTo);

    bool decomposeArcToCubic(float angle, float rx, float ry, FloatPoint&, FloatPoint&, bool largeArcFlag, bool sweepFlag);
    void parseClosePathSegment();
    bool parseMoveToSegment();
    bool parseLineToSegment();
    bool parseLineToHorizontalSegment();
    bool parseLineToVerticalSegment();
    bool parseCurveToCubicSegment();
    bool parseCurveToCubicSmoothSegment();
    bool parseCurveToQuadraticSegment();
    bool parseCurveToQuadraticSmoothSegment();
    bool parseArcToSegment();

    SingleThreadWeakRef<SVGPathSource> m_source;
    SingleThreadWeakRef<SVGPathConsumer> m_consumer;
    FloatPoint m_controlPoint;
    FloatPoint m_currentPoint;
    FloatPoint m_subPathPoint;
    PathCoordinateMode m_mode { AbsoluteCoordinates };
    const PathParsingMode m_pathParsingMode { NormalizedParsing };
    SVGPathSegType m_lastCommand { SVGPathSegType::Unknown };
    bool m_closePath { true };
};

} // namespace WebCore
