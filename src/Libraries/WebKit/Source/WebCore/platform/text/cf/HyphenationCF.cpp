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
#include "Hyphenation.h"

#include <wtf/Language.h>
#include <wtf/RetainPtr.h>
#include <wtf/TinyLRUCache.h>
#include <wtf/text/StringView.h>
#include <wtf/text/TextBreakIteratorInternalICU.h>

namespace WTF {

template<>
class TinyLRUCachePolicy<AtomString, RetainPtr<CFLocaleRef>>
{
public:
    static TinyLRUCache<AtomString, RetainPtr<CFLocaleRef>>& cache()
    {
        static NeverDestroyed<TinyLRUCache<AtomString, RetainPtr<CFLocaleRef>>> cache;
        return cache;
    }

    static bool isKeyNull(const AtomString& localeIdentifier)
    {
        return localeIdentifier.isNull();
    }

    static RetainPtr<CFLocaleRef> createValueForNullKey()
    {
        return nullptr;
    }

    static RetainPtr<CFLocaleRef> createValueForKey(const AtomString& localeIdentifier)
    {
        RetainPtr<CFLocaleRef> locale = adoptCF(CFLocaleCreate(kCFAllocatorDefault, localeIdentifier.string().createCFString().get()));

        return CFStringIsHyphenationAvailableForLocale(locale.get()) ? locale : nullptr;
    }

    static AtomString createKeyForStorage(const AtomString& key) { return key; }
};
}

namespace WebCore {

bool canHyphenate(const AtomString& localeIdentifier)
{
    if (localeIdentifier.isNull())
        return false;
    return TinyLRUCachePolicy<AtomString, RetainPtr<CFLocaleRef>>::cache().get(localeIdentifier);
}

size_t lastHyphenLocation(StringView text, size_t beforeIndex, const AtomString& localeIdentifier)
{
    RetainPtr<CFLocaleRef> locale = TinyLRUCachePolicy<AtomString, RetainPtr<CFLocaleRef>>::cache().get(localeIdentifier);

    CFOptionFlags searchAcrossWordBoundaries = 1;
    CFIndex result = CFStringGetHyphenationLocationBeforeIndex(text.createCFStringWithoutCopying().get(), beforeIndex, CFRangeMake(0, text.length()), searchAcrossWordBoundaries, locale.get(), nullptr);
    return result == kCFNotFound ? 0 : result;
}

} // namespace WebCore
