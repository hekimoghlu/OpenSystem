/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 7, 2024.
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
#include "SVGPathStringViewSource.h"

#include "SVGParserUtilities.h"

namespace WebCore {

SVGPathStringViewSource::SVGPathStringViewSource(StringView view)
    : m_is8BitSource(view.is8Bit())
{
    ASSERT(!view.isEmpty());

    if (m_is8BitSource)
        m_buffer8 = view.span8();
    else
        m_buffer16 = view.span16();
}

bool SVGPathStringViewSource::hasMoreData() const
{
    if (m_is8BitSource)
        return m_buffer8.hasCharactersRemaining();
    return m_buffer16.hasCharactersRemaining();
}

bool SVGPathStringViewSource::moveToNextToken()
{
    if (m_is8BitSource)
        return skipOptionalSVGSpaces(m_buffer8);
    return skipOptionalSVGSpaces(m_buffer16);
}

template <typename CharacterType> static std::optional<SVGPathSegType> nextCommandHelper(StringParsingBuffer<CharacterType>& buffer, SVGPathSegType previousCommand)
{
    // Check for remaining coordinates in the current command.
    if ((*buffer == '+' || *buffer == '-' || *buffer == '.' || isASCIIDigit(*buffer))
        && previousCommand != SVGPathSegType::ClosePath) {
        if (previousCommand == SVGPathSegType::MoveToAbs)
            return SVGPathSegType::LineToAbs;
        if (previousCommand == SVGPathSegType::MoveToRel)
            return SVGPathSegType::LineToRel;
        return previousCommand;
    }

    return std::nullopt;
}

SVGPathSegType SVGPathStringViewSource::nextCommand(SVGPathSegType previousCommand)
{
    if (m_is8BitSource) {
        if (auto nextCommand = nextCommandHelper(m_buffer8, previousCommand))
            return *nextCommand;
    } else {
        if (auto nextCommand = nextCommandHelper(m_buffer16, previousCommand))
            return *nextCommand;
    }

    return *parseSVGSegmentType();
}

template<typename F> decltype(auto) SVGPathStringViewSource::parse(F&& functor)
{
    if (m_is8BitSource)
        return functor(m_buffer8);
    return functor(m_buffer16);
}

std::optional<SVGPathSegType> SVGPathStringViewSource::parseSVGSegmentType()
{
    return parse([](auto& buffer) -> SVGPathSegType {
        auto character = *buffer;
        buffer++;
        switch (character) {
        case 'Z':
        case 'z':
            return SVGPathSegType::ClosePath;
        case 'M':
            return SVGPathSegType::MoveToAbs;
        case 'm':
            return SVGPathSegType::MoveToRel;
        case 'L':
            return SVGPathSegType::LineToAbs;
        case 'l':
            return SVGPathSegType::LineToRel;
        case 'C':
            return SVGPathSegType::CurveToCubicAbs;
        case 'c':
            return SVGPathSegType::CurveToCubicRel;
        case 'Q':
            return SVGPathSegType::CurveToQuadraticAbs;
        case 'q':
            return SVGPathSegType::CurveToQuadraticRel;
        case 'A':
            return SVGPathSegType::ArcAbs;
        case 'a':
            return SVGPathSegType::ArcRel;
        case 'H':
            return SVGPathSegType::LineToHorizontalAbs;
        case 'h':
            return SVGPathSegType::LineToHorizontalRel;
        case 'V':
            return SVGPathSegType::LineToVerticalAbs;
        case 'v':
            return SVGPathSegType::LineToVerticalRel;
        case 'S':
            return SVGPathSegType::CurveToCubicSmoothAbs;
        case 's':
            return SVGPathSegType::CurveToCubicSmoothRel;
        case 'T':
            return SVGPathSegType::CurveToQuadraticSmoothAbs;
        case 't':
            return SVGPathSegType::CurveToQuadraticSmoothRel;
        default:
            return SVGPathSegType::Unknown;
        }
    });
}

std::optional<SVGPathSource::MoveToSegment> SVGPathStringViewSource::parseMoveToSegment(FloatPoint)
{
    return parse([](auto& buffer) -> std::optional<MoveToSegment> {
        auto targetPoint = parseFloatPoint(buffer);
        if (!targetPoint)
            return std::nullopt;
        
        MoveToSegment segment;
        segment.targetPoint = WTFMove(*targetPoint);
        return segment;
    });
}

std::optional<SVGPathSource::LineToSegment> SVGPathStringViewSource::parseLineToSegment(FloatPoint)
{
    return parse([](auto& buffer) -> std::optional<LineToSegment> {
        auto targetPoint = parseFloatPoint(buffer);
        if (!targetPoint)
            return std::nullopt;
        
        LineToSegment segment;
        segment.targetPoint = WTFMove(*targetPoint);
        return segment;
    });
}

std::optional<SVGPathSource::LineToHorizontalSegment> SVGPathStringViewSource::parseLineToHorizontalSegment(FloatPoint)
{
    return parse([](auto& buffer) -> std::optional<LineToHorizontalSegment> {
        auto x = parseNumber(buffer);
        if (!x)
            return std::nullopt;
        
        LineToHorizontalSegment segment;
        segment.x = *x;
        return segment;
    });
}

std::optional<SVGPathSource::LineToVerticalSegment> SVGPathStringViewSource::parseLineToVerticalSegment(FloatPoint)
{
    return parse([](auto& buffer) -> std::optional<LineToVerticalSegment> {
        auto y = parseNumber(buffer);
        if (!y)
            return std::nullopt;
        
        LineToVerticalSegment segment;
        segment.y = *y;
        return segment;
    });
}

std::optional<SVGPathSource::CurveToCubicSegment> SVGPathStringViewSource::parseCurveToCubicSegment(FloatPoint)
{
    return parse([](auto& buffer) -> std::optional<CurveToCubicSegment> {
        auto point1 = parseFloatPoint(buffer);
        if (!point1)
            return std::nullopt;

        auto point2 = parseFloatPoint(buffer);
        if (!point2)
            return std::nullopt;

        auto targetPoint = parseFloatPoint(buffer);
        if (!targetPoint)
            return std::nullopt;

        CurveToCubicSegment segment;
        segment.point1 = *point1;
        segment.point2 = *point2;
        segment.targetPoint = *targetPoint;
        return segment;
    });
}

std::optional<SVGPathSource::CurveToCubicSmoothSegment> SVGPathStringViewSource::parseCurveToCubicSmoothSegment(FloatPoint)
{
    return parse([](auto& buffer) -> std::optional<CurveToCubicSmoothSegment> {
        auto point2 = parseFloatPoint(buffer);
        if (!point2)
            return std::nullopt;

        auto targetPoint = parseFloatPoint(buffer);
        if (!targetPoint)
            return std::nullopt;

        CurveToCubicSmoothSegment segment;
        segment.point2 = *point2;
        segment.targetPoint = *targetPoint;
        return segment;
    });
}

std::optional<SVGPathSource::CurveToQuadraticSegment> SVGPathStringViewSource::parseCurveToQuadraticSegment(FloatPoint)
{
    return parse([](auto& buffer) -> std::optional<CurveToQuadraticSegment> {
        auto point1 = parseFloatPoint(buffer);
        if (!point1)
            return std::nullopt;

        auto targetPoint = parseFloatPoint(buffer);
        if (!targetPoint)
            return std::nullopt;

        CurveToQuadraticSegment segment;
        segment.point1 = *point1;
        segment.targetPoint = *targetPoint;
        return segment;
    });
}

std::optional<SVGPathSource::CurveToQuadraticSmoothSegment> SVGPathStringViewSource::parseCurveToQuadraticSmoothSegment(FloatPoint)
{
    return parse([](auto& buffer) -> std::optional<CurveToQuadraticSmoothSegment> {
        auto targetPoint = parseFloatPoint(buffer);
        if (!targetPoint)
            return std::nullopt;

        CurveToQuadraticSmoothSegment segment;
        segment.targetPoint = *targetPoint;
        return segment;
    });
}

std::optional<SVGPathSource::ArcToSegment> SVGPathStringViewSource::parseArcToSegment(FloatPoint)
{
    return parse([](auto& buffer) -> std::optional<ArcToSegment> {
        auto rx = parseNumber(buffer);
        if (!rx)
            return std::nullopt;
        auto ry = parseNumber(buffer);
        if (!ry)
            return std::nullopt;
        auto angle = parseNumber(buffer);
        if (!angle)
            return std::nullopt;
        auto largeArc = parseArcFlag(buffer);
        if (!largeArc)
            return std::nullopt;
        auto sweep = parseArcFlag(buffer);
        if (!sweep)
            return std::nullopt;
        auto targetPoint = parseFloatPoint(buffer);
        if (!targetPoint)
            return std::nullopt;

        ArcToSegment segment;
        segment.rx = *rx;
        segment.ry = *ry;
        segment.angle = *angle;
        segment.largeArc = *largeArc;
        segment.sweep = *sweep;
        segment.targetPoint = *targetPoint;
        return segment;
    });
}

} // namespace WebKit
