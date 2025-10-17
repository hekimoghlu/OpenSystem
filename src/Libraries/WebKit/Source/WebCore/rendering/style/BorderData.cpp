/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 22, 2024.
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
#include "BorderData.h"

#include "OutlineValue.h"
#include "RenderStyle.h"
#include <wtf/PointerComparison.h>
#include <wtf/text/TextStream.h>

namespace WebCore {

bool BorderData::isEquivalentForPainting(const BorderData& other, bool currentColorDiffers) const
{
    if (!arePointingToEqualData(this, &other))
        return false;

    if (!currentColorDiffers)
        return true;

    auto visibleBorderHasCurrentColor = (m_top.isVisible() && m_top.color().containsCurrentColor())
        || (m_right.isVisible() && m_right.color().containsCurrentColor())
        || (m_bottom.isVisible() && m_bottom.color().containsCurrentColor())
        || (m_left.isVisible() && m_left.color().containsCurrentColor());
    return !visibleBorderHasCurrentColor;
}

TextStream& operator<<(TextStream& ts, const BorderValue& borderValue)
{
    ts << borderValue.width() << " " << borderValue.style() << " " << borderValue.color();
    return ts;
}

TextStream& operator<<(TextStream& ts, const OutlineValue& outlineValue)
{
    ts << static_cast<const BorderValue&>(outlineValue);
    ts.dumpProperty("outline-offset", outlineValue.offset());
    return ts;
}

void BorderData::dump(TextStream& ts, DumpStyleValues behavior) const
{
    if (behavior == DumpStyleValues::All || left() != BorderValue())
        ts.dumpProperty("left", left());
    if (behavior == DumpStyleValues::All || right() != BorderValue())
        ts.dumpProperty("right", right());
    if (behavior == DumpStyleValues::All || top() != BorderValue())
        ts.dumpProperty("top", top());
    if (behavior == DumpStyleValues::All || bottom() != BorderValue())
        ts.dumpProperty("bottom", bottom());

    ts.dumpProperty("image", image());

    if (behavior == DumpStyleValues::All || !topLeftRadius().isZero())
        ts.dumpProperty("top-left", topLeftRadius());
    if (behavior == DumpStyleValues::All || !topRightRadius().isZero())
        ts.dumpProperty("top-right", topRightRadius());
    if (behavior == DumpStyleValues::All || !bottomLeftRadius().isZero())
        ts.dumpProperty("bottom-left", bottomLeftRadius());
    if (behavior == DumpStyleValues::All || !bottomRightRadius().isZero())
        ts.dumpProperty("bottom-right", bottomRightRadius());
}

TextStream& operator<<(TextStream& ts, const BorderData& borderData)
{
    borderData.dump(ts);
    return ts;
}

} // namespace WebCore
