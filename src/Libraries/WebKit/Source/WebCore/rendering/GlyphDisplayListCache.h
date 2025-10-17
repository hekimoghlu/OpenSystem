/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 25, 2025.
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

#include "DisplayList.h"
#include "FloatSizeHash.h"
#include "FontCascade.h"
#include "Logging.h"
#include "TextRun.h"
#include "TextRunHash.h"
#include <wtf/HashMap.h>
#include <wtf/MemoryPressureHandler.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakRef.h>

namespace WebCore {

class LegacyInlineTextBox;
struct PaintInfo;

namespace InlineDisplay {
struct Box;
}

class GlyphDisplayListCacheEntry : public RefCounted<GlyphDisplayListCacheEntry>, public CanMakeSingleThreadWeakPtr<GlyphDisplayListCacheEntry> {
    WTF_MAKE_TZONE_ALLOCATED(GlyphDisplayListCacheEntry);
    friend struct GlyphDisplayListCacheKeyTranslator;
    friend void add(Hasher&, const GlyphDisplayListCacheEntry&);
public:
    static Ref<GlyphDisplayListCacheEntry> create(std::unique_ptr<DisplayList::DisplayList>&& displayList, const TextRun& textRun, const FontCascade& font, GraphicsContext& context)
    {
        return adoptRef(*new GlyphDisplayListCacheEntry(WTFMove(displayList), textRun, font, context));
    }

    ~GlyphDisplayListCacheEntry();

    bool operator==(const GlyphDisplayListCacheEntry& other) const
    {
        return m_textRun == other.m_textRun
            && m_scaleFactor == other.m_scaleFactor
            && m_fontCascadeGeneration == other.m_fontCascadeGeneration
            && m_shouldSubpixelQuantizeFont == other.m_shouldSubpixelQuantizeFont;
    }

    DisplayList::DisplayList& displayList() { return *m_displayList.get(); }

private:
    GlyphDisplayListCacheEntry(std::unique_ptr<DisplayList::DisplayList>&& displayList, const TextRun& textRun, const FontCascade& font, GraphicsContext& context)
        : m_displayList(WTFMove(displayList))
        , m_textRun(textRun.isolatedCopy())
        , m_scaleFactor(context.scaleFactor())
        , m_fontCascadeGeneration(font.generation())
        , m_shouldSubpixelQuantizeFont(context.shouldSubpixelQuantizeFonts())
    {
        ASSERT(m_displayList.get());
    }

    std::unique_ptr<DisplayList::DisplayList> m_displayList;

    TextRun m_textRun;
    FloatSize m_scaleFactor;
    unsigned m_fontCascadeGeneration;
    bool m_shouldSubpixelQuantizeFont;
};

inline void add(Hasher& hasher, const GlyphDisplayListCacheEntry& entry)
{
    add(hasher, entry.m_textRun, entry.m_scaleFactor.width(), entry.m_scaleFactor.height(), entry.m_fontCascadeGeneration, entry.m_shouldSubpixelQuantizeFont);
}

struct GlyphDisplayListCacheEntryHash {
    static unsigned hash(const GlyphDisplayListCacheEntry* entry) { return computeHash(*entry); }
    static unsigned hash(const SingleThreadWeakRef<GlyphDisplayListCacheEntry>& entry) { return computeHash(entry.get()); }
    static bool equal(const SingleThreadWeakRef<GlyphDisplayListCacheEntry>& a, const SingleThreadWeakRef<GlyphDisplayListCacheEntry>& b) { return a.ptr() == b.ptr(); }
    static bool equal(const SingleThreadWeakRef<GlyphDisplayListCacheEntry>& a, const GlyphDisplayListCacheEntry* b) { return a.ptr() == b; }
    static bool equal(const GlyphDisplayListCacheEntry* a, const SingleThreadWeakRef<GlyphDisplayListCacheEntry>& b) { return a == b.ptr(); }
    static constexpr bool safeToCompareToEmptyOrDeleted = false;
};

class GlyphDisplayListCache {
    WTF_MAKE_TZONE_ALLOCATED(GlyphDisplayListCache);
    friend class GlyphDisplayListCacheEntry;
public:
    GlyphDisplayListCache() = default;

    static GlyphDisplayListCache& singleton();

    DisplayList::DisplayList* get(const LegacyInlineTextBox&, const FontCascade&, GraphicsContext&, const TextRun&, const PaintInfo&);
    DisplayList::DisplayList* get(const InlineDisplay::Box&, const FontCascade&, GraphicsContext&, const TextRun&, const PaintInfo&);

    DisplayList::DisplayList* getIfExists(const LegacyInlineTextBox&);
    DisplayList::DisplayList* getIfExists(const InlineDisplay::Box&);

    void remove(const LegacyInlineTextBox& run) { remove(&run); }
    void remove(const InlineDisplay::Box& run) { remove(&run); }

    void clear();
    unsigned size() const;

    void setForceUseGlyphDisplayListForTesting(bool flag)
    {
        m_forceUseGlyphDisplayListForTesting = flag;
    }

private:
    static bool canShareDisplayList(const DisplayList::DisplayList&);

    template<typename LayoutRun> DisplayList::DisplayList* getDisplayList(const LayoutRun&, const FontCascade&, GraphicsContext&, const TextRun&, const PaintInfo&);
    template<typename LayoutRun> DisplayList::DisplayList* getIfExistsImpl(const LayoutRun&);
    void remove(const void* run);

    UncheckedKeyHashMap<const void*, Ref<GlyphDisplayListCacheEntry>> m_entriesForLayoutRun;
    UncheckedKeyHashSet<SingleThreadWeakRef<GlyphDisplayListCacheEntry>> m_entries;
    bool m_forceUseGlyphDisplayListForTesting { false };
};

} // namespace WebCore

namespace WTF {

template<> struct DefaultHash<SingleThreadWeakRef<WebCore::GlyphDisplayListCacheEntry>> : WebCore::GlyphDisplayListCacheEntryHash { };

} // namespace WTF
