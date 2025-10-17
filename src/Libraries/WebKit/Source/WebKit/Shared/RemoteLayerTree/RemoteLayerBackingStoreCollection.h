/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 13, 2021.
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

#import <WebCore/Timer.h>
#import <wtf/HashSet.h>
#import <wtf/Noncopyable.h>
#import <wtf/OptionSet.h>
#import <wtf/TZoneMalloc.h>
#import <wtf/WeakHashSet.h>
#import <wtf/WeakPtr.h>

namespace WebCore {
class ImageBuffer;
class ThreadSafeImageBufferFlusher;
enum class SetNonVolatileResult : uint8_t;
}

namespace WebKit {

class RemoteLayerBackingStore;
class RemoteLayerWithRemoteRenderingBackingStore;
class RemoteLayerWithInProcessRenderingBackingStore;
class RemoteLayerTreeContext;
class RemoteLayerTreeTransaction;
class RemoteImageBufferSetProxy;
class ThreadSafeImageBufferSetFlusher;

enum class BufferInSetType : uint8_t;
enum class SwapBuffersDisplayRequirement : uint8_t;

class RemoteLayerBackingStoreCollection : public CanMakeWeakPtr<RemoteLayerBackingStoreCollection> {
    WTF_MAKE_TZONE_ALLOCATED(RemoteLayerBackingStoreCollection);
    WTF_MAKE_NONCOPYABLE(RemoteLayerBackingStoreCollection);
public:
    RemoteLayerBackingStoreCollection(RemoteLayerTreeContext&);
    virtual ~RemoteLayerBackingStoreCollection();

    void ref() const;
    void deref() const;

    void markFrontBufferVolatileForTesting(RemoteLayerBackingStore&);
    virtual void backingStoreWasCreated(RemoteLayerBackingStore&);
    virtual void backingStoreWillBeDestroyed(RemoteLayerBackingStore&);

    void purgeFrontBufferForTesting(RemoteLayerBackingStore&);
    void purgeBackBufferForTesting(RemoteLayerBackingStore&);

    // Return value indicates whether the backing store needs to be included in the transaction.
    bool backingStoreWillBeDisplayed(RemoteLayerBackingStore&);
    bool backingStoreWillBeDisplayedWithRenderingSuppression(RemoteLayerBackingStore&);
    void backingStoreBecameUnreachable(RemoteLayerBackingStore&);

    virtual void prepareBackingStoresForDisplay(RemoteLayerTreeTransaction&);
    virtual bool paintReachableBackingStoreContents();

    void willFlushLayers();
    void willBuildTransaction();
    void willCommitLayerTree(RemoteLayerTreeTransaction&);
    Vector<std::unique_ptr<ThreadSafeImageBufferSetFlusher>> didFlushLayers(RemoteLayerTreeTransaction&);

    virtual void tryMarkAllBackingStoreVolatile(CompletionHandler<void(bool)>&&);

    void scheduleVolatilityTimer();

    virtual void gpuProcessConnectionWasDestroyed();

    RemoteLayerTreeContext& layerTreeContext() const;

protected:

    enum class VolatilityMarkingBehavior : uint8_t {
        IgnoreReachability              = 1 << 0,
        ConsiderTimeSinceLastDisplay    = 1 << 1,
    };

    virtual void markBackingStoreVolatileAfterReachabilityChange(RemoteLayerBackingStore&);
    virtual void markAllBackingStoreVolatileFromTimer();

    bool collectRemoteRenderingBackingStoreBufferIdentifiersToMarkVolatile(RemoteLayerWithRemoteRenderingBackingStore&, OptionSet<VolatilityMarkingBehavior>, MonotonicTime now, Vector<std::pair<Ref<RemoteImageBufferSetProxy>, OptionSet<BufferInSetType>>>&);

    bool collectAllRemoteRenderingBufferIdentifiersToMarkVolatile(OptionSet<VolatilityMarkingBehavior> liveBackingStoreMarkingBehavior, OptionSet<VolatilityMarkingBehavior> unparentedBackingStoreMarkingBehavior, Vector<std::pair<Ref<RemoteImageBufferSetProxy>, OptionSet<BufferInSetType>>>&);


private:
    bool markInProcessBackingStoreVolatile(RemoteLayerWithInProcessRenderingBackingStore&, OptionSet<VolatilityMarkingBehavior> = { }, MonotonicTime = { });
    bool markAllBackingStoreVolatile(OptionSet<VolatilityMarkingBehavior> liveBackingStoreMarkingBehavior, OptionSet<VolatilityMarkingBehavior> unparentedBackingStoreMarkingBehavior);

    bool updateUnreachableBackingStores();
    void volatilityTimerFired();

protected:
    void sendMarkBuffersVolatile(Vector<std::pair<Ref<RemoteImageBufferSetProxy>, OptionSet<BufferInSetType>>>&&, CompletionHandler<void(bool)>&&, bool forcePurge = false);

    static constexpr auto volatileBackingStoreAgeThreshold = 1_s;
    static constexpr auto volatileSecondaryBackingStoreAgeThreshold = 200_ms;

    WeakRef<RemoteLayerTreeContext> m_layerTreeContext;

    WeakHashSet<RemoteLayerBackingStore> m_liveBackingStore;
    WeakHashSet<RemoteLayerBackingStore> m_unparentedBackingStore;

    // Only used during a single flush.
    WeakHashSet<RemoteLayerBackingStore> m_reachableBackingStoreInLatestFlush;
    WeakHashSet<RemoteLayerBackingStore> m_backingStoresNeedingDisplay;

    WebCore::Timer m_volatilityTimer;

    bool m_inLayerFlush { false };
};

} // namespace WebKit
