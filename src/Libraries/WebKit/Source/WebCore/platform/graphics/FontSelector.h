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
#pragma once

#include "FontRanges.h"
#include <wtf/Forward.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>

namespace WTF {
class TextStream;
}

namespace WebCore {

class FontCache;
class FontCascadeDescription;
class FontDescription;
class FontSelectorClient;

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(FontAccessor);

class FontAccessor : public RefCounted<FontAccessor> {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(FontAccessor);
public:
    virtual ~FontAccessor() = default;

    virtual const Font* font(ExternalResourceDownloadPolicy) const = 0;
    virtual bool isLoading() const = 0;
};

class FontSelector : public RefCountedAndCanMakeWeakPtr<FontSelector> {
public:
    virtual ~FontSelector() = default;

    virtual FontRanges fontRangesForFamily(const FontDescription&, const AtomString&) = 0;
    virtual RefPtr<Font> fallbackFontAt(const FontDescription&, size_t) = 0;

    virtual size_t fallbackFontCount() = 0;

    virtual void opportunisticallyStartFontDataURLLoading(const FontCascadeDescription&, const AtomString& family) = 0;

    virtual void fontCacheInvalidated() { }

    virtual void registerForInvalidationCallbacks(FontSelectorClient&) = 0;
    virtual void unregisterForInvalidationCallbacks(FontSelectorClient&) = 0;

    virtual unsigned uniqueId() const = 0;
    virtual unsigned version() const = 0;

    virtual bool isSimpleFontSelectorForDescription() const = 0;

    virtual bool isCSSFontSelector() const { return false; }

};

WTF::TextStream& operator<<(WTF::TextStream&, const FontSelector&);

}
