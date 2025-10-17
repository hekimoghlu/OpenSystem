/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 2, 2024.
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
#include "SystemFallbackFontCache.h"

#include "FontCache.h"
#include "FontCascade.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/unicode/CharacterNames.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(SystemFallbackFontCache);

SystemFallbackFontCache& SystemFallbackFontCache::forCurrentThread()
{
    return FontCache::forCurrentThread().systemFallbackFontCache();
}

SystemFallbackFontCache* SystemFallbackFontCache::forCurrentThreadIfExists()
{
    auto* cache = FontCache::forCurrentThreadIfExists();
    if (!cache)
        return nullptr;

    return &cache->systemFallbackFontCache();
}

RefPtr<Font> SystemFallbackFontCache::systemFallbackFontForCharacterCluster(const Font* font, StringView characterCluster, const FontDescription& description, ResolvedEmojiPolicy resolvedEmojiPolicy, IsForPlatformFont isForPlatformFont)
{
    auto fontAddResult = m_characterFallbackMaps.add(font, CharacterFallbackMap());

    auto key = CharacterFallbackMapKey { description.computedLocale(), characterCluster.toString(), isForPlatformFont != IsForPlatformFont::No, resolvedEmojiPolicy };
    return fontAddResult.iterator->value.ensure(WTFMove(key), [&] {
        StringBuilder stringBuilder;
        stringBuilder.append(FontCascade::normalizeSpaces(characterCluster));

        // FIXME: Is this the right place to add the variation selectors?
        // Should this be done in platform-specific code instead?
        // The fact that Core Text accepts this information in the form of variation selectors
        // seems like a platform-specific quirk.
        // However, if we do this later in platform-specific code, we'd have to reallocate
        // the array and copy its contents, which seems wasteful.
        switch (resolvedEmojiPolicy) {
        case ResolvedEmojiPolicy::NoPreference:
            break;
        case ResolvedEmojiPolicy::RequireText:
            stringBuilder.append(textVariationSelector);
            break;
        case ResolvedEmojiPolicy::RequireEmoji:
            stringBuilder.append(emojiVariationSelector);
            break;
        }

        auto fallbackFont = FontCache::forCurrentThread().systemFallbackForCharacterCluster(description, *font, isForPlatformFont, FontCache::PreferColoredFont::No, stringBuilder).get();
        if (fallbackFont)
            fallbackFont->setIsUsedInSystemFallbackFontCache();
        return fallbackFont;
    }).iterator->value;
}

void SystemFallbackFontCache::remove(Font* font)
{
    m_characterFallbackMaps.remove(font);

    if (!font->isUsedInSystemFallbackFontCache())
        return;

    for (auto& characterMap : m_characterFallbackMaps.values()) {
        Vector<CharacterFallbackMapKey, 512> toRemove;
        for (auto& entry : characterMap) {
            if (entry.value == font)
                toRemove.append(entry.key);
        }
        for (auto& key : toRemove)
            characterMap.remove(key);
    }
}

} // namespace WebCore
