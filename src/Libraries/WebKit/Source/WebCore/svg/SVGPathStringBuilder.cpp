/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 21, 2024.
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
#include "SVGPathStringBuilder.h"

#include <wtf/text/WTFString.h>

namespace WebCore {

SVGPathStringBuilder::SVGPathStringBuilder() = default;

SVGPathStringBuilder::~SVGPathStringBuilder() = default;

String SVGPathStringBuilder::result()
{
    unsigned size = m_stringBuilder.length();
    if (!size)
        return String();

    // Remove trailing space.
    m_stringBuilder.shrink(size - 1);
    return m_stringBuilder.toString();
}

void SVGPathStringBuilder::incrementPathSegmentCount()
{
}

bool SVGPathStringBuilder::continueConsuming()
{
    return true;
}

static void appendFlag(StringBuilder& stringBuilder, bool flag)
{
    stringBuilder.append(flag ? '1' : '0', ' ');
}

static void appendNumber(StringBuilder& stringBuilder, float number)
{
    // FIXME: Shortest form would be better, but fixed precision is required for now to smooth over precision errors caused by converting float to double and back in CGPath on Cocoa platforms.
    stringBuilder.append(FormattedNumber::fixedPrecision(number), ' ');
}

static void appendPoint(StringBuilder& stringBuilder, const FloatPoint& point)
{
    appendNumber(stringBuilder, point.x());
    appendNumber(stringBuilder, point.y());
}

void SVGPathStringBuilder::moveTo(const FloatPoint& targetPoint, bool, PathCoordinateMode mode)
{
    if (mode == AbsoluteCoordinates)
        m_stringBuilder.append("M "_s);
    else
        m_stringBuilder.append("m "_s);

    appendPoint(m_stringBuilder, targetPoint);
}

void SVGPathStringBuilder::lineTo(const FloatPoint& targetPoint, PathCoordinateMode mode)
{
    if (mode == AbsoluteCoordinates)
        m_stringBuilder.append("L "_s);
    else
        m_stringBuilder.append("l "_s);

    appendPoint(m_stringBuilder, targetPoint);
}

void SVGPathStringBuilder::lineToHorizontal(float x, PathCoordinateMode mode)
{
    if (mode == AbsoluteCoordinates)
        m_stringBuilder.append("H "_s);
    else
        m_stringBuilder.append("h "_s);

    appendNumber(m_stringBuilder, x);
}

void SVGPathStringBuilder::lineToVertical(float y, PathCoordinateMode mode)
{
    if (mode == AbsoluteCoordinates)
        m_stringBuilder.append("V "_s);
    else
        m_stringBuilder.append("v "_s);

    appendNumber(m_stringBuilder, y);
}

void SVGPathStringBuilder::curveToCubic(const FloatPoint& point1, const FloatPoint& point2, const FloatPoint& targetPoint, PathCoordinateMode mode)
{
    if (mode == AbsoluteCoordinates)
        m_stringBuilder.append("C "_s);
    else
        m_stringBuilder.append("c "_s);

    appendPoint(m_stringBuilder, point1);
    appendPoint(m_stringBuilder, point2);
    appendPoint(m_stringBuilder, targetPoint);
}

void SVGPathStringBuilder::curveToCubicSmooth(const FloatPoint& point2, const FloatPoint& targetPoint, PathCoordinateMode mode)
{
    if (mode == AbsoluteCoordinates)
        m_stringBuilder.append("S "_s);
    else
        m_stringBuilder.append("s "_s);

    appendPoint(m_stringBuilder, point2);
    appendPoint(m_stringBuilder, targetPoint);
}

void SVGPathStringBuilder::curveToQuadratic(const FloatPoint& point1, const FloatPoint& targetPoint, PathCoordinateMode mode)
{
    if (mode == AbsoluteCoordinates)
        m_stringBuilder.append("Q "_s);
    else
        m_stringBuilder.append("q "_s);

    appendPoint(m_stringBuilder, point1);
    appendPoint(m_stringBuilder, targetPoint);
}

void SVGPathStringBuilder::curveToQuadraticSmooth(const FloatPoint& targetPoint, PathCoordinateMode mode)
{
    if (mode == AbsoluteCoordinates)
        m_stringBuilder.append("T "_s);
    else
        m_stringBuilder.append("t "_s);

    appendPoint(m_stringBuilder, targetPoint);
}

void SVGPathStringBuilder::arcTo(float r1, float r2, float angle, bool largeArcFlag, bool sweepFlag, const FloatPoint& targetPoint, PathCoordinateMode mode)
{
    if (mode == AbsoluteCoordinates)
        m_stringBuilder.append("A "_s);
    else
        m_stringBuilder.append("a "_s);

    appendNumber(m_stringBuilder, r1);
    appendNumber(m_stringBuilder, r2);
    appendNumber(m_stringBuilder, angle);
    appendFlag(m_stringBuilder, largeArcFlag);
    appendFlag(m_stringBuilder, sweepFlag);
    appendPoint(m_stringBuilder, targetPoint);
}

void SVGPathStringBuilder::closePath()
{
    m_stringBuilder.append("Z "_s);
}

} // namespace WebCore
