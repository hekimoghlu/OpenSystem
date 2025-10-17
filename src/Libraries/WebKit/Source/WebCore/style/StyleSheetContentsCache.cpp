/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 5, 2025.
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
#include "StyleSheetContentsCache.h"

#include "StyleSheetContents.h"

namespace WebCore {
namespace Style {

StyleSheetContentsCache::StyleSheetContentsCache() = default;

StyleSheetContentsCache& StyleSheetContentsCache::singleton()
{
    static NeverDestroyed<StyleSheetContentsCache> cache;
    return cache.get();
}

RefPtr<StyleSheetContents> StyleSheetContentsCache::get(const Key& key)
{
    return m_cache.get(key);
}

void StyleSheetContentsCache::add(Key&& key, Ref<StyleSheetContents> contents)
{
    ASSERT(contents->isCacheable());

    m_cache.add(WTFMove(key), contents);
    contents->addedToMemoryCache();

    static constexpr auto maximumCacheSize = 256;
    if (m_cache.size() > maximumCacheSize) {
        auto toRemove = m_cache.random();
        toRemove->value->removedFromMemoryCache();
        m_cache.remove(toRemove);
    }
}

void StyleSheetContentsCache::clear()
{
    m_cache.clear();
}

}
}
