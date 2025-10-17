/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 26, 2021.
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

#include "DisplayListItems.h"
#include "DisplayListResourceHeap.h"
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WTF {
class TextStream;
}

namespace WebCore {
namespace DisplayList {

class DisplayList {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(DisplayList, WEBCORE_EXPORT);
    WTF_MAKE_NONCOPYABLE(DisplayList);
public:
    DisplayList(OptionSet<ReplayOption> options = { })
        : m_options(options)
    {
    }

    WEBCORE_EXPORT void append(Item&&);
    void shrinkToFit();

    WEBCORE_EXPORT void clear();
    WEBCORE_EXPORT bool isEmpty() const;

    const Vector<Item>& items() const { return m_items; }
    Vector<Item>& items() { return m_items; }
    const ResourceHeap& resourceHeap() const { return m_resourceHeap; }

    void cacheImageBuffer(ImageBuffer&);
    void cacheNativeImage(NativeImage&);
    void cacheFont(Font&);
    void cacheDecomposedGlyphs(DecomposedGlyphs&);
    void cacheGradient(Gradient&);
    void cacheFilter(Filter&);

    WEBCORE_EXPORT String asText(OptionSet<AsTextFlag>) const;
    void dump(WTF::TextStream&) const;

    const OptionSet<ReplayOption>& replayOptions() const { return m_options; }

private:
    Vector<Item> m_items;
    ResourceHeap m_resourceHeap;
    OptionSet<ReplayOption> m_options;
};

WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, const DisplayList&);

} // DisplayList
} // WebCore
