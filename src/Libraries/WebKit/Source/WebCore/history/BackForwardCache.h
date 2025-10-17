/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 9, 2025.
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

#include "BackForwardItemIdentifier.h"
#include "HistoryItem.h"
#include <wtf/Forward.h>
#include <wtf/ListHashSet.h>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class CachedPage;
class LocalFrame;
class Page;

enum class PruningReason { None, ProcessSuspended, MemoryPressure, ReachedMaxSize };

class BackForwardCache {
    WTF_MAKE_TZONE_ALLOCATED(BackForwardCache);
    WTF_MAKE_NONCOPYABLE(BackForwardCache);
public:
    // Function to obtain the global back/forward cache.
    WEBCORE_EXPORT static BackForwardCache& singleton();

    bool canCache(Page&) const;

    // Used when memory is low to prune some cached pages.
    WEBCORE_EXPORT void pruneToSizeNow(unsigned maxSize, PruningReason);
    WEBCORE_EXPORT void setMaxSize(unsigned); // number of pages to cache.
    unsigned maxSize() const { return m_maxSize; }

    WEBCORE_EXPORT std::unique_ptr<CachedPage> suspendPage(Page&);
    WEBCORE_EXPORT bool addIfCacheable(HistoryItem&, Page*); // Prunes if maxSize() is exceeded.
    WEBCORE_EXPORT void remove(BackForwardItemIdentifier);
    WEBCORE_EXPORT void remove(HistoryItem&);
    CachedPage* get(HistoryItem&, Page*);
    std::unique_ptr<CachedPage> take(HistoryItem&, Page*);

    void removeAllItemsForPage(Page&);

    WEBCORE_EXPORT void clearEntriesForOrigins(const UncheckedKeyHashSet<RefPtr<SecurityOrigin>>&);

    unsigned pageCount() const { return m_items.size(); }
    WEBCORE_EXPORT unsigned frameCount() const;

    void markPagesForDeviceOrPageScaleChanged(Page&);
    void markPagesForContentsSizeChanged(Page&);
#if ENABLE(VIDEO)
    void markPagesForCaptionPreferencesChanged();
#endif

    bool isInBackForwardCache(BackForwardItemIdentifier) const;
    bool hasCachedPageExpired(BackForwardItemIdentifier) const;

private:
    BackForwardCache();
    ~BackForwardCache() = delete; // Make sure nobody accidentally calls delete -- WebCore does not delete singletons.

    static bool canCachePageContainingThisFrame(LocalFrame&);

    enum class ForceSuspension : bool { No, Yes };
    std::unique_ptr<CachedPage> trySuspendPage(Page&, ForceSuspension);
    void prune(PruningReason);
    void dump() const;

    HashMap<BackForwardItemIdentifier, std::variant<PruningReason, UniqueRef<CachedPage>>> m_cachedPageMap;
    ListHashSet<BackForwardItemIdentifier> m_items;
    unsigned m_maxSize {0};

#if ASSERT_ENABLED
    bool m_isInRemoveAllItemsForPage { false };
#endif

    friend class WTF::NeverDestroyed<BackForwardCache>;
};

} // namespace WebCore
