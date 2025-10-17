/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 31, 2023.
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

#include "TextFlags.h"
#include <wtf/HashMap.h>
#include <wtf/HashTraits.h>
#include <wtf/Hasher.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/AtomString.h>

namespace WebCore {

class Font;
class FontDescription;
enum class IsForPlatformFont : bool;
    
struct CharacterFallbackMapKey {
    AtomString locale;
    String string;
    bool isForPlatformFont { false };
    ResolvedEmojiPolicy resolvedEmojiPolicy { ResolvedEmojiPolicy::NoPreference };

    bool operator==(const CharacterFallbackMapKey& other) const = default;
};

inline void add(Hasher& hasher, const CharacterFallbackMapKey& key)
{
    add(hasher, key.locale, key.string, key.isForPlatformFont, key.resolvedEmojiPolicy);
}

struct CharacterFallbackMapKeyHash {
    static unsigned hash(const CharacterFallbackMapKey& key) { return computeHash(key); }
    static bool equal(const CharacterFallbackMapKey& a, const CharacterFallbackMapKey& b) { return a == b; }
    static const bool safeToCompareToEmptyOrDeleted = false;
};

class SystemFallbackFontCache {
    WTF_MAKE_TZONE_ALLOCATED(SystemFallbackFontCache);
    WTF_MAKE_NONCOPYABLE(SystemFallbackFontCache);
public:
    static SystemFallbackFontCache& forCurrentThread();
    static SystemFallbackFontCache* forCurrentThreadIfExists();

    SystemFallbackFontCache() = default;

    RefPtr<Font> systemFallbackFontForCharacterCluster(const Font*, StringView, const FontDescription&, ResolvedEmojiPolicy, IsForPlatformFont);
    void remove(Font*);

private:
    struct CharacterFallbackMapKeyHashTraits : SimpleClassHashTraits<CharacterFallbackMapKey> {
        static void constructDeletedValue(CharacterFallbackMapKey& slot) { new (NotNull, &slot) CharacterFallbackMapKey { { }, WTF::HashTableDeletedValue, { } }; }
        static bool isDeletedValue(const CharacterFallbackMapKey& key) { return key.string.isHashTableDeletedValue(); }
    };

    // Fonts are not ref'd to avoid cycles.
    // FIXME: Consider changing these maps to use WeakPtr instead of raw pointers.
    using CharacterFallbackMap = UncheckedKeyHashMap<CharacterFallbackMapKey, Font*, CharacterFallbackMapKeyHash, CharacterFallbackMapKeyHashTraits>;

    UncheckedKeyHashMap<const Font*, CharacterFallbackMap> m_characterFallbackMaps;
};

} // namespace WebCore
