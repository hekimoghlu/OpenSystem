/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 11, 2024.
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
#include <wtf/text/WTFString.h>

namespace WebCore {

class FloatPoint;
class Path;
class SVGPathByteStream;
class SVGPathSeg;
class SVGPathSegList;

// Path -> String
String buildStringFromPath(const Path&);

// String/SVGPathByteStream -> Path
Path buildPathFromString(StringView);
Path buildPathFromByteStream(const SVGPathByteStream&);

// SVGPathSegList/String -> SVGPathByteStream
bool buildSVGPathByteStreamFromSVGPathSegList(const SVGPathSegList&, SVGPathByteStream& result, PathParsingMode, bool checkForInitialMoveTo = true);
bool buildSVGPathByteStreamFromString(StringView, SVGPathByteStream&, PathParsingMode);

// SVGPathByteStream -> String
bool buildStringFromByteStream(const SVGPathByteStream&, String&, PathParsingMode, bool checkForInitialMoveTo = true);

// SVGPathByteStream -> SVGPathSegList
bool buildSVGPathSegListFromByteStream(const SVGPathByteStream&, SVGPathSegList&, PathParsingMode);

bool canBlendSVGPathByteStreams(const SVGPathByteStream& from, const SVGPathByteStream& to);

bool buildAnimatedSVGPathByteStream(const SVGPathByteStream& from, const SVGPathByteStream& to, SVGPathByteStream& result, float progress);
bool addToSVGPathByteStream(SVGPathByteStream& streamToAppendTo, const SVGPathByteStream& from, unsigned repeatCount = 1);

unsigned getSVGPathSegAtLengthFromSVGPathByteStream(const SVGPathByteStream&, float length);
float getTotalLengthOfSVGPathByteStream(const SVGPathByteStream&);
FloatPoint getPointAtLengthOfSVGPathByteStream(const SVGPathByteStream&, float length);

// Convert an SVG path byte stream containing a mixed of relative/absolute draw commands into another byte stream
// such that all draw commands are absolute. Returns nullptr if an error occurs.
std::optional<SVGPathByteStream> convertSVGPathByteStreamToAbsoluteCoordinates(const SVGPathByteStream&);

} // namespace WebCore
