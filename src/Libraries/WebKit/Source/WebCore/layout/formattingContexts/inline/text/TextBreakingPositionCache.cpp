/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 6, 2024.
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
#include "TextBreakingPositionCache.h"

#include "RenderStyle.h"
#include "RenderStyleInlines.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {
namespace Layout {

WTF_MAKE_TZONE_ALLOCATED_IMPL(TextBreakingPositionCache);

static constexpr size_t evictionSoftThreshold = 500000; // At this amount of content (string + breaking position list) we should start evicting
static constexpr size_t evictionHardCapMultiplier = 5; // Do not let the cache grow beyond this
static constexpr Seconds idleIntervalForEviction { 10_s };

TextBreakingPositionCache& TextBreakingPositionCache::singleton()
{
    static NeverDestroyed<TextBreakingPositionCache> cache;
    return cache.get();
}

TextBreakingPositionCache::TextBreakingPositionCache()
    : m_delayedEvictionTimer([this] { evict(); })
{
}

void TextBreakingPositionCache::evict()
{
    while (m_cachedContentSize > evictionSoftThreshold && !m_breakingPositionMap.isEmpty()) {
        auto evictedEntry = m_breakingPositionMap.random();
        m_cachedContentSize -= (std::get<0>(evictedEntry->key).length() + 4 * evictedEntry->value.size());
        m_breakingPositionMap.remove(evictedEntry->key);
    }
}

void TextBreakingPositionCache::set(const Key& key, List&& breakingPositionList)
{
    ASSERT(!m_breakingPositionMap.contains(key));

    auto evictIfNeeded = [&] {
        if (m_cachedContentSize < evictionSoftThreshold)
            return;

        ASSERT(!m_breakingPositionMap.isEmpty());
        auto isBelowHardThreshold = m_cachedContentSize < evictionSoftThreshold * evictionHardCapMultiplier;
        if (isBelowHardThreshold) {
            m_delayedEvictionTimer.startOneShot(idleIntervalForEviction);
            return;
        }
        evict();
    };
    evictIfNeeded();

    m_cachedContentSize += (std::get<0>(key).length() + 4 * breakingPositionList.size());
    m_breakingPositionMap.set(key, WTFMove(breakingPositionList));
}

const TextBreakingPositionCache::List* TextBreakingPositionCache::get(const Key& key) const
{
    auto iterator = m_breakingPositionMap.find(key);
    if (iterator == m_breakingPositionMap.end())
        return { };
    return &iterator->value;
}

void TextBreakingPositionCache::clear()
{
    m_breakingPositionMap.clear();
    m_cachedContentSize = 0;
}

void add(Hasher& hasher, const TextBreakingPositionContext& context)
{
    add(hasher, context.whitespaceCollapseBehavior, context.overflowWrap, context.lineBreak, context.wordBreak, context.nbspMode, context.locale);
}

}
}
