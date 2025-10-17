/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 15, 2024.
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
#include "IntSize.h"

#include "FloatSize.h"
#include <wtf/JSONValues.h>
#include <wtf/text/TextStream.h>

namespace WebCore {

IntSize::IntSize(const FloatSize& s)
    : m_width(clampToInteger(s.width()))
    , m_height(clampToInteger(s.height()))
{
}

IntSize IntSize::constrainedBetween(const IntSize& min, const IntSize& max) const
{
    return {
        std::max(min.width(), std::min(max.width(), m_width)),
        std::max(min.height(), std::min(max.height(), m_height))
    };
}

TextStream& operator<<(TextStream& ts, const IntSize& size)
{
    return ts << "width=" << size.width() << " height=" << size.height();
}

Ref<JSON::Object> IntSize::toJSONObject() const
{
    auto object = JSON::Object::create();

    object->setDouble("width"_s, m_width);
    object->setDouble("height"_s, m_height);

    return object;
}

String IntSize::toJSONString() const
{
    return toJSONObject()->toJSONString();
}

}
