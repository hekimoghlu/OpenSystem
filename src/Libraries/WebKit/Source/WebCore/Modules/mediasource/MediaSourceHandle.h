/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 25, 2025.
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
#if ENABLE(MEDIA_SOURCE_IN_WORKERS)

#include <wtf/Forward.h>
#include <wtf/Function.h>
#include <wtf/Lock.h>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class MediaSource;
class MediaSource;
class MediaSourcePrivate;
class MediaSourcePrivateClient;

class MediaSourceHandle
    : public RefCounted<MediaSourceHandle> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(MediaSourceHandle);
public:
    static Ref<MediaSourceHandle> create(Ref<MediaSourceHandle>&&);
    Ref<MediaSourceHandle> detach();

    virtual ~MediaSourceHandle();

    bool isDetached() const { return m_detached; }
    bool canDetach() const;
    bool detachable() const { return m_detachable; }

    void setHasEverBeenAssignedAsSrcObject();
    bool hasEverBeenAssignedAsSrcObject() const;
    bool isManaged() const;

    using TaskType = Function<void(MediaSource&)>;
    void ensureOnDispatcher(TaskType&&, bool forceRun = false) const;

    RefPtr<MediaSourcePrivateClient> mediaSourcePrivateClient() const;
    RefPtr<MediaSourcePrivate> mediaSourcePrivate() const;

private:
    class SharedPrivate;
    friend class MediaSource;

    using DispatcherType = Function<void(TaskType, bool)>;

    static Ref<MediaSourceHandle> create(MediaSource&, DispatcherType&&, bool);
    MediaSourceHandle(MediaSource&, DispatcherType&&, bool);
    explicit MediaSourceHandle(MediaSourceHandle&);
    void mediaSourceDidOpen(MediaSourcePrivate&);
    void setDetached(bool value) { m_detached = value; }

    const bool m_detachable;
    bool m_detached { false };
    Ref<SharedPrivate> m_private;
};

using DetachedMediaSourceHandle = MediaSourceHandle;

} // namespace WebCore

#endif // MEDIA_SOURCE_IN_WORKERS
