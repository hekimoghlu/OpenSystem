/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 27, 2025.
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

#include "FcUniquePtr.h"
#include "FontCascadeCache.h"
#include "FontDescription.h"
#include "RefPtrFontconfig.h"
#include <wtf/HashMap.h>
#include <wtf/HashTraits.h>
#include <wtf/Hasher.h>

namespace WebCore {

struct FontSetCacheKey {
    FontSetCacheKey() = default;

    FontSetCacheKey(const FontDescription& description, bool coloredFont)
        : descriptionKey(description)
        , preferColoredFont(coloredFont)
    {
    }

    explicit FontSetCacheKey(WTF::HashTableDeletedValueType deletedValue)
        : descriptionKey(deletedValue)
    {
    }

    bool operator==(const FontSetCacheKey& other) const
    {
        return descriptionKey == other.descriptionKey && preferColoredFont == other.preferColoredFont;
    }

    bool isHashTableDeletedValue() const { return descriptionKey.isHashTableDeletedValue(); }

    FontDescriptionKey descriptionKey;
    bool preferColoredFont { false };
};

inline void add(Hasher& hasher, const FontSetCacheKey& key)
{
    add(hasher, key.descriptionKey, key.preferColoredFont);
}

struct FontSetCacheKeyHash {
    static unsigned hash(const FontSetCacheKey& key) { return computeHash(key); }
    static bool equal(const FontSetCacheKey& a, const FontSetCacheKey& b) { return a == b; }
    static const bool safeToCompareToEmptyOrDeleted = true;
};

class FontSetCache {
    WTF_MAKE_NONCOPYABLE(FontSetCache);
public:
    FontSetCache() = default;

    RefPtr<FcPattern> bestForCharacters(const FontDescription&, bool, StringView);
    void clear();

private:
    struct FontSet {
        WTF_MAKE_STRUCT_FAST_ALLOCATED;

        explicit FontSet(RefPtr<FcPattern>&&);

        RefPtr<FcPattern> pattern;
        FcUniquePtr<FcFontSet> fontSet;
        Vector<std::pair<FcPattern*, FcCharSet*>> patterns;
    };

    UncheckedKeyHashMap<FontSetCacheKey, std::unique_ptr<FontSet>, FontSetCacheKeyHash, SimpleClassHashTraits<FontSetCacheKey>> m_cache;
};

} // namespace WebCore
