/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 26, 2023.
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

#include "FloatRect.h"
#include "Timer.h"

#include <CoreGraphics/CoreGraphics.h>
#include <wtf/CheckedPtr.h>
#include <wtf/HashCountedSet.h>
#include <wtf/HashSet.h>
#include <wtf/HashTraits.h>
#include <wtf/Lock.h>
#include <wtf/RetainPtr.h>
#include <wtf/RunLoop.h>
#include <wtf/TZoneMalloc.h>

#define CACHE_SUBIMAGES 1

namespace WebCore {

#if CACHE_SUBIMAGES

class CGSubimageCacheWithTimer final : public CanMakeThreadSafeCheckedPtr<CGSubimageCacheWithTimer> {
    WTF_MAKE_TZONE_ALLOCATED(CGSubimageCacheWithTimer);
    WTF_MAKE_NONCOPYABLE(CGSubimageCacheWithTimer);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(CGSubimageCacheWithTimer);
public:
    struct CacheEntry {
        RetainPtr<CGImageRef> image;
        RetainPtr<CGImageRef> subimage;
        FloatRect rect;
        MonotonicTime lastAccessTime;
    };

    struct CacheEntryTraits : HashTraits<CacheEntry> {
        typedef HashTraits<RetainPtr<CGImageRef>> ImageTraits;

        static const bool emptyValueIsZero = true;

        static const bool hasIsEmptyValueFunction = true;
        static bool isEmptyValue(const CacheEntry& value) { return !value.image; }

        static void constructDeletedValue(CacheEntry& slot) { ImageTraits::constructDeletedValue(slot.image); }
        static bool isDeletedValue(const CacheEntry& value) { return ImageTraits::isDeletedValue(value.image); }
    };

    struct CacheHash {
        static unsigned hash(CGImageRef image, const FloatRect& rect)
        {
            return pairIntHash(PtrHash<CGImageRef>::hash(image),
                (static_cast<unsigned>(rect.x()) << 16) | static_cast<unsigned>(rect.y()));
        }
        static unsigned hash(const CacheEntry& key)
        {
            return hash(key.image.get(), key.rect);
        }
        static bool equal(const CacheEntry& a, const CacheEntry& b)
        {
            return a.image == b.image && a.rect == b.rect;
        }
        static const bool safeToCompareToEmptyOrDeleted = true;
    };

    static RetainPtr<CGImageRef> getSubimage(CGImageRef, const FloatRect&);
    static void clearImage(CGImageRef);
    static void clear();

private:
    static constexpr Seconds cachePruneDelay { 500_ms };
    static constexpr Seconds cacheEntryLifetime { 500_ms };
    static constexpr int maxCacheSize = 300;

    typedef UncheckedKeyHashSet<CacheEntry, CacheHash, CacheEntryTraits> CacheHashSet;

    CGSubimageCacheWithTimer();
    void pruneCacheTimerFired();

    RetainPtr<CGImageRef> subimage(CGImageRef, const FloatRect&);
    void clearImageAndSubimages(CGImageRef);
    void prune() WTF_REQUIRES_LOCK(m_lock);
    void clearAll();

    Lock m_lock;
    HashCountedSet<CGImageRef> m_imageCounts WTF_GUARDED_BY_LOCK(m_lock);
    CacheHashSet m_cache WTF_GUARDED_BY_LOCK(m_lock);
    RunLoop::Timer m_timer WTF_GUARDED_BY_LOCK(m_lock);

    static CGSubimageCacheWithTimer& subimageCache();
    static bool subimageCacheExists();
    static CGSubimageCacheWithTimer* s_cache;
};

#endif // CACHE_SUBIMAGES

}
