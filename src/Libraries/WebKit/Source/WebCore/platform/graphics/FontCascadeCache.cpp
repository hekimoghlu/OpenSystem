/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 2, 2022.
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
#include "FontCascadeCache.h"

#include "CSSFontSelector.h"
#include "FontCache.h"
#include "FontCascadeDescription.h"
#include <wtf/RefPtr.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(FontDescriptionKeyRareData);
WTF_MAKE_TZONE_ALLOCATED_IMPL(FontCascadeCache);

FontFamilyName::FontFamilyName() = default;

FontFamilyName::FontFamilyName(const AtomString& name)
    : m_name { name }
{
}

const AtomString& FontFamilyName::string() const
{
    return m_name;
}

void add(Hasher& hasher, const FontFamilyName& name)
{
    // FIXME: Would be better to hash the characters in the name instead of hashing a hash.
    if (!name.string().isNull())
        add(hasher, FontCascadeDescription::familyNameHash(name.string()));
}

bool operator==(const FontFamilyName& a, const FontFamilyName& b)
{
    return (a.string().isNull() || b.string().isNull()) ? a.string() == b.string() : FontCascadeDescription::familyNamesAreEqual(a.string(), b.string());
}

FontCascadeCache& FontCascadeCache::forCurrentThread()
{
    return FontCache::forCurrentThread().fontCascadeCache();
}

void FontCascadeCache::invalidate()
{
    m_entries.clear();
}

void FontCascadeCache::clearWidthCaches()
{
    for (auto& value : m_entries.values())
        value->fonts.get().widthCache().clear();
}

void FontCascadeCache::pruneUnreferencedEntries()
{
    m_entries.removeIf([](auto& entry) {
        return entry.value->fonts.get().hasOneRef();
    });
}

void FontCascadeCache::pruneSystemFallbackFonts()
{
    for (auto& entry : m_entries.values())
        entry->fonts->pruneSystemFallbacks();
}

static FontCascadeCacheKey makeFontCascadeCacheKey(const FontCascadeDescription& description, FontSelector* fontSelector)
{
    unsigned familyCount = description.familyCount();
    auto hasComplexFontSelector = fontSelector && !fontSelector->isSimpleFontSelectorForDescription();
    return FontCascadeCacheKey {
        FontDescriptionKey(description),
        Vector<FontFamilyName, 3>(familyCount, [&](size_t i) { return description.familyAt(i); }),
        hasComplexFontSelector ? fontSelector->uniqueId() : 0,
        hasComplexFontSelector ? fontSelector->version() : 0,
        hasComplexFontSelector
    };
}

Ref<FontCascadeFonts> FontCascadeCache::retrieveOrAddCachedFonts(const FontCascadeDescription& fontDescription, FontSelector* fontSelector)
{
    auto key = makeFontCascadeCacheKey(fontDescription, fontSelector);
    auto addResult = m_entries.add(key, nullptr);
    if (!addResult.isNewEntry)
        return addResult.iterator->value->fonts.get();

    auto& newEntry = addResult.iterator->value;
    newEntry = makeUnique<FontCascadeCacheEntry>(FontCascadeCacheEntry { WTFMove(key), FontCascadeFonts::create() });
    Ref<FontCascadeFonts> glyphs = newEntry->fonts.get();


#if !PLATFORM(IOS_FAMILY)
    static constexpr unsigned unreferencedPruneInterval = 1000;
    static constexpr int maximumEntries = 5000;
#else
    static constexpr unsigned unreferencedPruneInterval = 50;
    static constexpr int maximumEntries = 400;
#endif
    static unsigned pruneCounter;
    // Referenced FontCascadeFonts would exist anyway so pruning them saves little memory.
    if (!(++pruneCounter % unreferencedPruneInterval))
        pruneUnreferencedEntries();
    // Prevent pathological growth.
    if (m_entries.size() > maximumEntries)
        m_entries.remove(m_entries.random());
    return glyphs;
}

} // namespace WebCore
