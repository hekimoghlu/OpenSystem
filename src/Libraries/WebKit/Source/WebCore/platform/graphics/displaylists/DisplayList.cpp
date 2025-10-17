/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 8, 2021.
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
#include "DisplayList.h"

#include "DecomposedGlyphs.h"
#include "DisplayListResourceHeap.h"
#include "Filter.h"
#include "Font.h"
#include "ImageBuffer.h"
#include "Logging.h"
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/TextStream.h>

namespace WebCore {
namespace DisplayList {

WTF_MAKE_TZONE_ALLOCATED_IMPL(DisplayList);

void DisplayList::append(Item&& item)
{
    m_items.append(WTFMove(item));
}

void DisplayList::shrinkToFit()
{
    m_items.shrinkToFit();
}

void DisplayList::clear()
{
    m_items.clear();
    m_resourceHeap.clearAllResources();
}

bool DisplayList::isEmpty() const
{
    return m_items.isEmpty();
}

void DisplayList::cacheImageBuffer(ImageBuffer& imageBuffer)
{
    m_resourceHeap.add(Ref { imageBuffer });
}

void DisplayList::cacheNativeImage(NativeImage& image)
{
    m_resourceHeap.add(Ref { image });
}

void DisplayList::cacheFont(Font& font)
{
    m_resourceHeap.add(Ref { font });
}

void DisplayList::cacheDecomposedGlyphs(DecomposedGlyphs& decomposedGlyphs)
{
    m_resourceHeap.add(Ref { decomposedGlyphs });
}

void DisplayList::cacheGradient(Gradient& gradient)
{
    m_resourceHeap.add(Ref { gradient });
}

void DisplayList::cacheFilter(Filter& filter)
{
    m_resourceHeap.add(Ref { filter });
}

String DisplayList::asText(OptionSet<AsTextFlag> flags) const
{
    TextStream stream(TextStream::LineMode::MultipleLine, TextStream::Formatting::SVGStyleRect);
    for (const auto& item : m_items) {
        if (!shouldDumpItem(item, flags))
            continue;

        TextStream::GroupScope group(stream);
        dumpItem(stream, item, flags);
    }
    return stream.release();
}

void DisplayList::dump(TextStream& ts) const
{
    TextStream::GroupScope group(ts);
    ts << "display list";

    for (const auto& item : m_items) {
        TextStream::GroupScope group(ts);
        dumpItem(ts, item, { AsTextFlag::IncludePlatformOperations, AsTextFlag::IncludeResourceIdentifiers });
    }
}

TextStream& operator<<(TextStream& ts, const DisplayList& displayList)
{
    displayList.dump(ts);
    return ts;
}

} // namespace DisplayList
} // namespace WebCore
