/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 29, 2023.
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
#include "FloatSize.h"

#include "FloatConversion.h"
#include "IntSize.h"
#include <limits>
#include <math.h>
#include <wtf/JSONValues.h>
#include <wtf/text/TextStream.h>

namespace WebCore {

FloatSize FloatSize::constrainedBetween(const FloatSize& min, const FloatSize& max) const
{
    return {
        std::max(min.width(), std::min(max.width(), m_width)),
        std::max(min.height(), std::min(max.height(), m_height))
    };
}

bool FloatSize::isExpressibleAsIntSize() const
{
    return isWithinIntRange(m_width) && isWithinIntRange(m_height);
}

FloatSize FloatSize::narrowPrecision(double width, double height)
{
    return FloatSize(narrowPrecisionToFloat(width), narrowPrecisionToFloat(height));
}

TextStream& operator<<(TextStream& ts, const FloatSize& size)
{
    return ts << "width=" << TextStream::FormatNumberRespectingIntegers(size.width())
        << " height=" << TextStream::FormatNumberRespectingIntegers(size.height());
}

Ref<JSON::Object> FloatSize::toJSONObject() const
{
    auto object = JSON::Object::create();

    object->setDouble("width"_s, m_width);
    object->setDouble("height"_s, m_height);

    return object;
}

String FloatSize::toJSONString() const
{
    return toJSONObject()->toJSONString();
}

}
