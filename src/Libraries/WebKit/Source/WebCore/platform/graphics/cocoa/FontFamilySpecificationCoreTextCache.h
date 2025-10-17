/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 17, 2024.
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

#include "FontCascadeCache.h"
#include <CoreText/CoreText.h>
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

struct FontFamilySpecificationKey {
    RetainPtr<CTFontDescriptorRef> fontDescriptor;
    FontDescriptionKey fontDescriptionKey;

    FontFamilySpecificationKey() = default;

    FontFamilySpecificationKey(CTFontDescriptorRef fontDescriptor, const FontDescription& fontDescription)
        : fontDescriptor(fontDescriptor)
        , fontDescriptionKey(fontDescription)
    { }

    explicit FontFamilySpecificationKey(WTF::HashTableDeletedValueType deletedValue)
        : fontDescriptionKey(deletedValue)
    { }

    bool operator==(const FontFamilySpecificationKey& other) const
    {
        return safeCFEqual(fontDescriptor.get(), other.fontDescriptor.get()) && fontDescriptionKey == other.fontDescriptionKey;
    }

    bool isHashTableDeletedValue() const { return fontDescriptionKey.isHashTableDeletedValue(); }
};

inline void add(Hasher& hasher, const FontFamilySpecificationKey& key)
{
    // FIXME: Ideally, we wouldn't be hashing a hash.
    add(hasher, safeCFHash(key.fontDescriptor.get()), key.fontDescriptionKey);
}

struct FontFamilySpecificationKeyHash {
    static unsigned hash(const FontFamilySpecificationKey& key) { return computeHash(key); }
    static bool equal(const FontFamilySpecificationKey& a, const FontFamilySpecificationKey& b) { return a == b; }
    static const bool safeToCompareToEmptyOrDeleted = true;
};

class FontFamilySpecificationCoreTextCache {
    WTF_MAKE_TZONE_ALLOCATED(FontFamilySpecificationCoreTextCache);
    WTF_MAKE_NONCOPYABLE(FontFamilySpecificationCoreTextCache);
public:
    FontFamilySpecificationCoreTextCache() = default;

    static FontFamilySpecificationCoreTextCache& forCurrentThread();

    template<typename Functor> FontPlatformData& ensure(FontFamilySpecificationKey&&, Functor&&);
    void clear();

private:
    UncheckedKeyHashMap<FontFamilySpecificationKey, std::unique_ptr<FontPlatformData>, FontFamilySpecificationKeyHash, SimpleClassHashTraits<FontFamilySpecificationKey>> m_fonts;
};

template<typename Functor> FontPlatformData& FontFamilySpecificationCoreTextCache::ensure(FontFamilySpecificationKey&& key, Functor&& functor)
{
    auto& fontPlatformData = m_fonts.ensure(std::forward<FontFamilySpecificationKey>(key), std::forward<Functor>(functor)).iterator->value;
    ASSERT(fontPlatformData);
    return *fontPlatformData;
}

} // namespace WebCore
