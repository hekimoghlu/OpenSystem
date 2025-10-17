/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 19, 2021.
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

#if HAVE(IOSURFACE)

#include "IOSurface.h"
#include "IntSize.h"
#include "IntSizeHash.h"
#include "Timer.h"
#include <wtf/Deque.h>
#include <wtf/HashMap.h>
#include <wtf/Lock.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/RunLoop.h>
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeRefCounted.h>

namespace WebCore {

class DestinatationColorSpace;

class IOSurfacePool : public ThreadSafeRefCounted<IOSurfacePool> {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(IOSurfacePool, WEBCORE_EXPORT);
    WTF_MAKE_NONCOPYABLE(IOSurfacePool);
    friend class LazyNeverDestroyed<IOSurfacePool>;

public:
    WEBCORE_EXPORT static IOSurfacePool& sharedPoolSingleton();
    WEBCORE_EXPORT static Ref<IOSurfacePool> create();

    WEBCORE_EXPORT ~IOSurfacePool();

    std::unique_ptr<IOSurface> takeSurface(IntSize, const DestinationColorSpace&, IOSurface::Format);
    WEBCORE_EXPORT void addSurface(std::unique_ptr<IOSurface>&&);

    WEBCORE_EXPORT void discardAllSurfaces();

    WEBCORE_EXPORT void setPoolSize(size_t);

private:
    IOSurfacePool();

    struct CachedSurfaceDetails {
        CachedSurfaceDetails()
            : hasMarkedPurgeable(false)
        { }

        void resetLastUseTime() { lastUseTime = MonotonicTime::now(); }

        MonotonicTime lastUseTime;
        bool hasMarkedPurgeable;
    };

    typedef Deque<std::unique_ptr<IOSurface>> CachedSurfaceQueue;
    typedef UncheckedKeyHashMap<IntSize, CachedSurfaceQueue> CachedSurfaceMap;
    typedef UncheckedKeyHashMap<IOSurface*, CachedSurfaceDetails> CachedSurfaceDetailsMap;

#if PLATFORM(MAC)
    static constexpr size_t defaultMaximumBytesCached { 256 * MB };
#else
    static constexpr size_t defaultMaximumBytesCached { 64 * MB };
#endif

    // We'll never allow more than 1/2 of the cache to be filled with in-use surfaces, because
    // they can't be immediately returned when requested (but will be freed up in the future).
    static constexpr size_t maximumInUseBytes = defaultMaximumBytesCached / 2;
    
    bool shouldCacheSurface(const IOSurface&) const WTF_REQUIRES_LOCK(m_lock);

    void willAddSurface(IOSurface&, bool inUse) WTF_REQUIRES_LOCK(m_lock);
    void didRemoveSurface(IOSurface&, bool inUse) WTF_REQUIRES_LOCK(m_lock);
    void didUseSurfaceOfSize(IntSize) WTF_REQUIRES_LOCK(m_lock);

    void insertSurfaceIntoPool(std::unique_ptr<IOSurface>) WTF_REQUIRES_LOCK(m_lock);

    void evict(size_t additionalSize) WTF_REQUIRES_LOCK(m_lock);
    void tryEvictInUseSurface() WTF_REQUIRES_LOCK(m_lock);
    void tryEvictOldestCachedSurface() WTF_REQUIRES_LOCK(m_lock);

    void scheduleCollectionTimer() WTF_REQUIRES_LOCK(m_lock);
    void collectionTimerFired();
    void collectInUseSurfaces() WTF_REQUIRES_LOCK(m_lock);
    bool markOlderSurfacesPurgeable() WTF_REQUIRES_LOCK(m_lock);

    void platformGarbageCollectNow();

    void discardAllSurfacesInternal() WTF_REQUIRES_LOCK(m_lock);

    String poolStatistics() const WTF_REQUIRES_LOCK(m_lock);

    Lock m_lock;
    RunLoop::Timer m_collectionTimer WTF_GUARDED_BY_LOCK(m_lock);
    CachedSurfaceMap m_cachedSurfaces WTF_GUARDED_BY_LOCK(m_lock);
    CachedSurfaceQueue m_inUseSurfaces WTF_GUARDED_BY_LOCK(m_lock);
    CachedSurfaceDetailsMap m_surfaceDetails WTF_GUARDED_BY_LOCK(m_lock);
    Vector<IntSize> m_sizesInPruneOrder WTF_GUARDED_BY_LOCK(m_lock);

    size_t m_bytesCached WTF_GUARDED_BY_LOCK(m_lock) { 0 };
    size_t m_inUseBytesCached WTF_GUARDED_BY_LOCK(m_lock) { 0 };
    size_t m_maximumBytesCached WTF_GUARDED_BY_LOCK(m_lock) { defaultMaximumBytesCached };
};

}
#endif // HAVE(IOSURFACE)
