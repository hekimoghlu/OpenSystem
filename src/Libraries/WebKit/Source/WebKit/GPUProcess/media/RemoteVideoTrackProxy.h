/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 12, 2023.
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

#if ENABLE(GPU_PROCESS) && ENABLE(VIDEO)

#include "MessageReceiver.h"
#include <WebCore/MediaPlayerIdentifier.h>
#include <WebCore/TrackBase.h>
#include <WebCore/VideoTrackPrivate.h>
#include <wtf/Ref.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeWeakPtr.h>
#include <wtf/WeakPtr.h>

namespace IPC {
class Connection;
class Decoder;
}

namespace WebKit {

class GPUConnectionToWebProcess;
struct VideoTrackPrivateRemoteConfiguration;

class RemoteVideoTrackProxy final
    : public ThreadSafeRefCounted<RemoteVideoTrackProxy, WTF::DestructionThread::Main>
    , private WebCore::VideoTrackPrivateClient {
    WTF_MAKE_TZONE_ALLOCATED(RemoteVideoTrackProxy);
public:
    static Ref<RemoteVideoTrackProxy> create(GPUConnectionToWebProcess& connectionToWebProcess, WebCore::VideoTrackPrivate& trackPrivate, WebCore::MediaPlayerIdentifier mediaPlayerIdentifier)
    {
        return adoptRef(*new RemoteVideoTrackProxy(connectionToWebProcess, trackPrivate, mediaPlayerIdentifier));
    }

    virtual ~RemoteVideoTrackProxy();

    WebCore::TrackID id() const { return m_trackPrivate->id(); };
    void setSelected(bool selected)
    {
        m_selected = selected;
        Ref { m_trackPrivate }->setSelected(selected);
    }
    bool operator==(const WebCore::VideoTrackPrivate& track) const { return track == m_trackPrivate.get(); }

private:
    RemoteVideoTrackProxy(GPUConnectionToWebProcess&, WebCore::VideoTrackPrivate&, WebCore::MediaPlayerIdentifier);

    // VideoTrackPrivateClient
    void selectedChanged(bool) final;
    void configurationChanged(const WebCore::PlatformVideoTrackConfiguration&) final { updateConfiguration(); }

    // TrackPrivateBaseClient
    void idChanged(WebCore::TrackID) final;
    void labelChanged(const AtomString&) final;
    void languageChanged(const AtomString&) final;
    void willRemove() final;

    VideoTrackPrivateRemoteConfiguration configuration();
    void updateConfiguration();

    ThreadSafeWeakPtr<GPUConnectionToWebProcess> m_connectionToWebProcess;
    Ref<WebCore::VideoTrackPrivate> m_trackPrivate;
    WebCore::TrackID m_id;
    WebCore::MediaPlayerIdentifier m_mediaPlayerIdentifier;
    bool m_selected { false };
    size_t m_clientRegistrationId { 0 };
};

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS) && ENABLE(VIDEO)
