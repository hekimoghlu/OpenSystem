/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 23, 2022.
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
#include "MediaSourceHandle.h"

#if ENABLE(MEDIA_SOURCE_IN_WORKERS)

#include "MediaSource.h"
#include "MediaSourcePrivate.h"
#include "MediaSourcePrivateClient.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/ThreadSafeWeakPtr.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(MediaSourceHandle);

class MediaSourceHandle::SharedPrivate final : public ThreadSafeRefCounted<SharedPrivate> {
public:
    static Ref<SharedPrivate> create(MediaSource& mediaSource, MediaSourceHandle::DispatcherType&& dispatcher) { return adoptRef(*new SharedPrivate(mediaSource, WTFMove(dispatcher))); }

    void dispatch(MediaSourceHandle::TaskType&& task, bool forceRunInWorker = false) const
    {
        m_dispatcher(WTFMove(task), forceRunInWorker);
    }

    void setMediaSourcePrivate(MediaSourcePrivate& mediaSourcePrivate)
    {
        Locker locker { m_lock };
        m_private = mediaSourcePrivate;
    }

    RefPtr<MediaSourcePrivate> mediaSourcePrivate() const
    {
        Locker locker { m_lock };
        return m_private.get();
    }

    friend class MediaSourceHandle;
    SharedPrivate(MediaSource& mediaSource, MediaSourceHandle::DispatcherType&& dispatcher)
        : m_client(mediaSource.client().get())
        , m_isManaged(mediaSource.isManaged())
        , m_dispatcher(WTFMove(dispatcher))
    {
    }
    mutable Lock m_lock;
    ThreadSafeWeakPtr<MediaSourcePrivate> m_private WTF_GUARDED_BY_LOCK(m_lock);
    std::atomic<bool> m_hasEverBeenAssignedAsSrcObject { false };
    ThreadSafeWeakPtr<MediaSourcePrivateClient> m_client;
    const bool m_isManaged;
    const MediaSourceHandle::DispatcherType m_dispatcher;
};

Ref<MediaSourceHandle> MediaSourceHandle::create(MediaSource& mediaSource, MediaSourceHandle::DispatcherType&& dispatcher, bool detachable)
{
    return adoptRef(*new MediaSourceHandle(mediaSource, WTFMove(dispatcher), detachable));
}

Ref<MediaSourceHandle> MediaSourceHandle::create(Ref<MediaSourceHandle>&& other)
{
    other->setDetached(false);
    return other;
}

MediaSourceHandle::MediaSourceHandle(MediaSource& mediaSource, MediaSourceHandle::DispatcherType&& dispatcher, bool detachable)
    : m_detachable(detachable)
    , m_private(MediaSourceHandle::SharedPrivate::create(mediaSource, WTFMove(dispatcher)))
{
}

MediaSourceHandle::MediaSourceHandle(MediaSourceHandle& other)
    : m_detachable(other.m_detachable)
    , m_private(other.m_private)
{
    ASSERT(!other.m_detached);
    other.m_detached = true;
}

MediaSourceHandle::~MediaSourceHandle() = default;

bool MediaSourceHandle::canDetach() const
{
    return !isDetached() && !m_private->m_hasEverBeenAssignedAsSrcObject;
}

void MediaSourceHandle::setHasEverBeenAssignedAsSrcObject()
{
    m_private->m_hasEverBeenAssignedAsSrcObject = true;
}

bool MediaSourceHandle::hasEverBeenAssignedAsSrcObject() const
{
    return m_private->m_hasEverBeenAssignedAsSrcObject;
}

bool MediaSourceHandle::isManaged() const
{
    return m_private->m_isManaged;
}

void MediaSourceHandle::ensureOnDispatcher(MediaSourceHandle::TaskType&& task, bool forceRun) const
{
    m_private->dispatch(WTFMove(task), forceRun);
}

Ref<DetachedMediaSourceHandle> MediaSourceHandle::detach()
{
    return adoptRef(*new MediaSourceHandle(*this));
}

RefPtr<MediaSourcePrivateClient> MediaSourceHandle::mediaSourcePrivateClient() const
{
    return m_private->m_client.get();
}

void MediaSourceHandle::mediaSourceDidOpen(MediaSourcePrivate& privateMediaSource)
{
    m_private->setMediaSourcePrivate(privateMediaSource);
}

RefPtr<MediaSourcePrivate> MediaSourceHandle::mediaSourcePrivate() const
{
    return m_private->mediaSourcePrivate();
}

} // namespace WebCore

#endif // MEDIA_SOURCE_IN_WORKERS
