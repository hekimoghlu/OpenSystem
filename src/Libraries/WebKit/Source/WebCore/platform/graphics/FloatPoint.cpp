/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 11, 2023.
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
#include "FloatPoint.h"

#include "AffineTransform.h"
#include "FloatConversion.h"
#include "FloatRect.h"
#include "IntPoint.h"
#include "TransformationMatrix.h"
#include <limits>
#include <math.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/TextStream.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(FloatPoint);

FloatPoint::FloatPoint(const IntPoint& p) : m_x(p.x()), m_y(p.y())
{
}

FloatPoint FloatPoint::constrainedBetween(const FloatPoint& min, const FloatPoint& max) const
{
    return {
        std::max(min.x(), std::min(max.x(), m_x)),
        std::max(min.y(), std::min(max.y(), m_y))
    };
}

FloatPoint FloatPoint::constrainedWithin(const FloatRect& rect) const
{
    return constrainedBetween(rect.minXMinYCorner(), rect.maxXMaxYCorner());
}

void FloatPoint::normalize()
{
    float tempLength = length();

    if (tempLength) {
        m_x /= tempLength;
        m_y /= tempLength;
    }
}

float FloatPoint::slopeAngleRadians() const
{
    return atan2f(m_y, m_x);
}

FloatPoint FloatPoint::matrixTransform(const AffineTransform& transform) const
{
    double newX, newY;
    transform.map(static_cast<double>(m_x), static_cast<double>(m_y), newX, newY);
    return narrowPrecision(newX, newY);
}

FloatPoint FloatPoint::matrixTransform(const TransformationMatrix& transform) const
{
    double newX, newY;
    transform.map(static_cast<double>(m_x), static_cast<double>(m_y), newX, newY);
    return narrowPrecision(newX, newY);
}

FloatPoint FloatPoint::narrowPrecision(double x, double y)
{
    return FloatPoint(narrowPrecisionToFloat(x), narrowPrecisionToFloat(y));
}

TextStream& operator<<(TextStream& ts, const FloatPoint& p)
{
    // FIXME: callers should use the NumberRespectingIntegers flag.
    return ts << "(" << TextStream::FormatNumberRespectingIntegers(p.x()) << "," << TextStream::FormatNumberRespectingIntegers(p.y()) << ")";
}

Ref<JSON::Object> FloatPoint::toJSONObject() const
{
    auto object = JSON::Object::create();

    object->setDouble("x"_s, m_x);
    object->setDouble("y"_s, m_y);

    return object;
}

String FloatPoint::toJSONString() const
{
    return toJSONObject()->toJSONString();
}

}
