/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 19, 2021.
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

#include "Font.h"
#include "FontCascadeDescription.h"
#include "FontRanges.h"
#include "FontSelector.h"
#include "GlyphPage.h"
#include "WidthCache.h"
#include <wtf/EnumeratedArray.h>
#include <wtf/Forward.h>
#include <wtf/HashFunctions.h>
#include <wtf/HashTraits.h>
#include <wtf/MainThread.h>
#include <wtf/TriState.h>

#if PLATFORM(IOS_FAMILY)
#include "WebCoreThread.h"
#endif

namespace WTF {
class TextStream;
}

namespace WebCore {

class FontPlatformData;
class FontSelector;
class GraphicsContext;
class IntRect;
class MixedFontGlyphPage;

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(FontCascadeFonts);
class FontCascadeFonts : public RefCounted<FontCascadeFonts> {
    WTF_MAKE_NONCOPYABLE(FontCascadeFonts);
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(FontCascadeFonts);
public:
    static Ref<FontCascadeFonts> create() { return adoptRef(*new FontCascadeFonts()); }
    static Ref<FontCascadeFonts> createForPlatformFont(const FontPlatformData& platformData) { return adoptRef(*new FontCascadeFonts(platformData)); }

    WEBCORE_EXPORT ~FontCascadeFonts();

    bool isForPlatformFont() const { return m_isForPlatformFont; }

    GlyphData glyphDataForCharacter(char32_t, const FontCascadeDescription&, FontSelector*, FontVariant, ResolvedEmojiPolicy);

    bool isFixedPitch(const FontCascadeDescription&, FontSelector*);

    bool canTakeFixedPitchFastContentMeasuring(const FontCascadeDescription&, FontSelector*);

    bool isLoadingCustomFonts() const;

    // FIXME: It should be possible to combine fontSelectorVersion and generation.
    unsigned generation() const { return m_generation; }

    WidthCache& widthCache() { return m_widthCache; }
    const WidthCache& widthCache() const { return m_widthCache; }

    const Font& primaryFont(const FontCascadeDescription&, FontSelector*);
    WEBCORE_EXPORT const FontRanges& realizeFallbackRangesAt(const FontCascadeDescription&, FontSelector*, unsigned fallbackIndex);

    void pruneSystemFallbacks();

private:
    FontCascadeFonts();
    FontCascadeFonts(const FontPlatformData&);

    GlyphData glyphDataForSystemFallback(char32_t, const FontCascadeDescription&, FontSelector*, FontVariant, ResolvedEmojiPolicy, bool systemFallbackShouldBeInvisible);
    GlyphData glyphDataForVariant(char32_t, const FontCascadeDescription&, FontSelector*, FontVariant, ResolvedEmojiPolicy, unsigned fallbackIndex = 0);

    WEBCORE_EXPORT void determinePitch(const FontCascadeDescription&, FontSelector*);
    WEBCORE_EXPORT void determineCanTakeFixedPitchFastContentMeasuring(const FontCascadeDescription&, FontSelector*);

    Vector<FontRanges, 1> m_realizedFallbackRanges;
    unsigned m_lastRealizedFallbackIndex { 0 };

    class GlyphPageCacheEntry {
    public:
        GlyphPageCacheEntry() = default;
        GlyphPageCacheEntry(RefPtr<GlyphPage>&&);

        GlyphData glyphDataForCharacter(char32_t);

        void setSingleFontPage(RefPtr<GlyphPage>&&);
        void setGlyphDataForCharacter(char32_t, GlyphData);

        bool isNull() const { return !m_singleFont && !m_mixedFont; }
        bool isMixedFont() const { return !!m_mixedFont; }
    
    private:
        // Only one of these is non-null.
        RefPtr<GlyphPage> m_singleFont;
        std::unique_ptr<MixedFontGlyphPage> m_mixedFont;
    };

    EnumeratedArray<ResolvedEmojiPolicy, UncheckedKeyHashMap<unsigned, GlyphPageCacheEntry, IntHash<unsigned>, WTF::UnsignedWithZeroKeyHashTraits<unsigned>>, ResolvedEmojiPolicy::RequireEmoji> m_cachedPages;

    UncheckedKeyHashSet<RefPtr<Font>> m_systemFallbackFontSet;

    SingleThreadWeakPtr<const Font> m_cachedPrimaryFont;

    WidthCache m_widthCache;

    unsigned short m_generation { 0 };
    Pitch m_pitch { UnknownPitch };
    bool m_isForPlatformFont { false };
    TriState m_canTakeFixedPitchFastContentMeasuring : 2 { TriState::Indeterminate };
#if ASSERT_ENABLED
    std::optional<Ref<Thread>> m_thread;
#endif
};

inline bool FontCascadeFonts::isFixedPitch(const FontCascadeDescription& description, FontSelector* fontSelector)
{
    if (m_pitch == UnknownPitch)
        determinePitch(description, fontSelector);
    return m_pitch == FixedPitch;
}

inline bool FontCascadeFonts::canTakeFixedPitchFastContentMeasuring(const FontCascadeDescription& description, FontSelector* fontSelector)
{
    if (m_canTakeFixedPitchFastContentMeasuring == TriState::Indeterminate)
        determineCanTakeFixedPitchFastContentMeasuring(description, fontSelector);
    return m_canTakeFixedPitchFastContentMeasuring == TriState::True;
}

inline const Font& FontCascadeFonts::primaryFont(const FontCascadeDescription& description, FontSelector* fontSelector)
{
    ASSERT(m_thread ? m_thread->ptr() == &Thread::current() : isMainThread());
    if (!m_cachedPrimaryFont) {
        auto& primaryRanges = realizeFallbackRangesAt(description, fontSelector, 0);
        m_cachedPrimaryFont = primaryRanges.glyphDataForCharacter(' ', ExternalResourceDownloadPolicy::Allow).font.get();
        if (!m_cachedPrimaryFont)
            m_cachedPrimaryFont = primaryRanges.rangeAt(0).font(ExternalResourceDownloadPolicy::Allow);
        else if (m_cachedPrimaryFont->isInterstitial()) {
            for (unsigned index = 1; ; ++index) {
                auto& localRanges = realizeFallbackRangesAt(description, fontSelector, index);
                if (localRanges.isNull())
                    break;
                WeakPtr font = localRanges.glyphDataForCharacter(' ', ExternalResourceDownloadPolicy::Forbid).font.get();
                if (font && !font->isInterstitial()) {
                    m_cachedPrimaryFont = WTFMove(font);
                    break;
                }
            }
        }
    }
    ASSERT(m_cachedPrimaryFont);
    return *m_cachedPrimaryFont;
}

WTF::TextStream& operator<<(WTF::TextStream&, const FontCascadeFonts&);

} // namespace WebCore
