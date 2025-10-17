/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 5, 2023.
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
#include "SVGPathUtilities.h"

#include "FloatPoint.h"
#include "Path.h"
#include "PathTraversalState.h"
#include "SVGPathAbsoluteConverter.h"
#include "SVGPathBlender.h"
#include "SVGPathBuilder.h"
#include "SVGPathByteStreamBuilder.h"
#include "SVGPathByteStreamSource.h"
#include "SVGPathConsumer.h"
#include "SVGPathElement.h"
#include "SVGPathParser.h"
#include "SVGPathSegListBuilder.h"
#include "SVGPathSegListSource.h"
#include "SVGPathStringBuilder.h"
#include "SVGPathStringViewSource.h"
#include "SVGPathTraversalStateBuilder.h"

namespace WebCore {

Path buildPathFromString(StringView d)
{
    if (d.isEmpty())
        return { };

    Path path;
    SVGPathBuilder builder(path);
    SVGPathStringViewSource source(d);
    SVGPathParser::parse(source, builder);
    return path;
}

String buildStringFromPath(const Path& path)
{
    StringBuilder builder;

    if (!path.isEmpty()) {
        path.applyElements([&builder] (const PathElement& element) {
            switch (element.type) {
            case PathElement::Type::MoveToPoint:
                builder.append('M', element.points[0].x(), ' ', element.points[0].y());
                break;
            case PathElement::Type::AddLineToPoint:
                builder.append('L', element.points[0].x(), ' ', element.points[0].y());
                break;
            case PathElement::Type::AddQuadCurveToPoint:
                builder.append('Q', element.points[0].x(), ' ', element.points[0].y(), ',', element.points[1].x(), ' ', element.points[1].y());
                break;
            case PathElement::Type::AddCurveToPoint:
                builder.append('C', element.points[0].x(), ' ', element.points[0].y(), ',', element.points[1].x(), ' ', element.points[1].y(), ',', element.points[2].x(), ' ', element.points[2].y());
                break;
            case PathElement::Type::CloseSubpath:
                builder.append('Z');
                break;
            }
        });
    }

    return builder.toString();
}

bool buildSVGPathByteStreamFromSVGPathSegList(const SVGPathSegList& list, SVGPathByteStream& stream, PathParsingMode parsingMode, bool checkForInitialMoveTo)
{
    stream.clear();
    if (list.isEmpty())
        return true;

    SVGPathSegListSource source(list);
    return SVGPathParser::parseToByteStream(source, stream, parsingMode, checkForInitialMoveTo);
}

Path buildPathFromByteStream(const SVGPathByteStream& stream)
{
    if (stream.isEmpty())
        return { };

    if (auto path = stream.cachedPath())
        return path.value();

    Path path;
    SVGPathBuilder builder(path);
    SVGPathByteStreamSource source(stream);
    SVGPathParser::parse(source, builder);
    stream.cachePath(path);
    return path;
}

bool buildSVGPathSegListFromByteStream(const SVGPathByteStream& stream, SVGPathSegList& list, PathParsingMode mode)
{
    if (stream.isEmpty())
        return true;

    SVGPathSegListBuilder builder(list);
    SVGPathByteStreamSource source(stream);
    return SVGPathParser::parse(source, builder, mode);
}

bool buildStringFromByteStream(const SVGPathByteStream& stream, String& result, PathParsingMode parsingMode, bool checkForInitialMoveTo)
{
    if (stream.isEmpty())
        return true;

    SVGPathByteStreamSource source(stream);
    return SVGPathParser::parseToString(source, result, parsingMode, checkForInitialMoveTo);
}

bool buildSVGPathByteStreamFromString(StringView d, SVGPathByteStream& result, PathParsingMode parsingMode)
{
    result.clear();
    if (d.isEmpty())
        return true;

    SVGPathStringViewSource source(d);
    return SVGPathParser::parseToByteStream(source, result, parsingMode);
}

bool canBlendSVGPathByteStreams(const SVGPathByteStream& fromStream, const SVGPathByteStream& toStream)
{
    SVGPathByteStreamSource fromSource(fromStream);
    SVGPathByteStreamSource toSource(toStream);
    return SVGPathBlender::canBlendPaths(fromSource, toSource);
}

bool buildAnimatedSVGPathByteStream(const SVGPathByteStream& fromStream, const SVGPathByteStream& toStream, SVGPathByteStream& result, float progress)
{
    ASSERT(&toStream != &result);
    result.clear();
    if (toStream.isEmpty())
        return true;

    SVGPathByteStreamBuilder builder(result);

    SVGPathByteStreamSource fromSource(fromStream);
    SVGPathByteStreamSource toSource(toStream);
    return SVGPathBlender::blendAnimatedPath(fromSource, toSource, builder, progress);
}

bool addToSVGPathByteStream(SVGPathByteStream& streamToAppendTo, const SVGPathByteStream& byStream, unsigned repeatCount)
{
    // The byStream will be blended with streamToAppendTo. So streamToAppendTo has to have elements.
    if (streamToAppendTo.isEmpty() || byStream.isEmpty())
        return true;

    // builder is the destination of blending fromSource and bySource. The stream of builder
    // (i.e. streamToAppendTo) has to be cleared before calling addAnimatedPath.
    SVGPathByteStreamBuilder builder(streamToAppendTo);

    SVGPathByteStream fromStreamCopy = WTFMove(streamToAppendTo);

    SVGPathByteStreamSource fromSource(fromStreamCopy);
    SVGPathByteStreamSource bySource(byStream);
    return SVGPathBlender::addAnimatedPath(fromSource, bySource, builder, repeatCount);
}

unsigned getSVGPathSegAtLengthFromSVGPathByteStream(const SVGPathByteStream& stream, float length)
{
    if (stream.isEmpty())
        return 0;

    PathTraversalState traversalState(PathTraversalState::Action::SegmentAtLength);
    SVGPathTraversalStateBuilder builder(traversalState, length);

    SVGPathByteStreamSource source(stream);
    SVGPathParser::parse(source, builder);
    return builder.pathSegmentIndex();
}

float getTotalLengthOfSVGPathByteStream(const SVGPathByteStream& stream)
{
    if (stream.isEmpty())
        return 0;

    PathTraversalState traversalState(PathTraversalState::Action::TotalLength);

    SVGPathTraversalStateBuilder builder(traversalState);

    SVGPathByteStreamSource source(stream);
    SVGPathParser::parse(source, builder);
    return builder.totalLength();
}

FloatPoint getPointAtLengthOfSVGPathByteStream(const SVGPathByteStream& stream, float length)
{
    if (stream.isEmpty())
        return { };

    PathTraversalState traversalState(PathTraversalState::Action::VectorAtLength);

    SVGPathTraversalStateBuilder builder(traversalState, length);

    SVGPathByteStreamSource source(stream);
    SVGPathParser::parse(source, builder);
    return builder.currentPoint();
}

std::optional<SVGPathByteStream> convertSVGPathByteStreamToAbsoluteCoordinates(const SVGPathByteStream& stream)
{
    SVGPathByteStream result;
    if (stream.isEmpty())
        return result;

    SVGPathByteStreamBuilder builder(result);
    SVGPathAbsoluteConverter converter(builder);

    SVGPathByteStreamSource source(stream);

    if (!SVGPathParser::parse(source, converter, UnalteredParsing, false))
        return std::nullopt;

    return result;
}

}
