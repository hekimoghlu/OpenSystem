/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 21, 2023.
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
#include <WebCore/AudioTrackPrivate.h>
#include <WebCore/MediaPlayerIdentifier.h>
#include <WebCore/TrackBase.h>
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
struct AudioTrackPrivateRemoteConfiguration;

class RemoteAudioTrackProxy final
    : public ThreadSafeRefCounted<RemoteAudioTrackProxy, WTF::DestructionThread::Main>
    , public WebCore::AudioTrackPrivateClient {
    WTF_MAKE_TZONE_ALLOCATED(RemoteAudioTrackProxy);
public:
    static Ref<RemoteAudioTrackProxy> create(GPUConnectionToWebProcess& connectionToWebProcess, WebCore::AudioTrackPrivate& trackPrivate, WebCore::MediaPlayerIdentifier mediaPlayerIdentifier)
    {
        return adoptRef(*new RemoteAudioTrackProxy(connectionToWebProcess, trackPrivate, mediaPlayerIdentifier));
    }

    virtual ~RemoteAudioTrackProxy();

    WebCore::TrackID id() const { return m_trackPrivate->id(); };
    void setEnabled(bool enabled)
    {
        m_enabled = enabled;
        Ref { m_trackPrivate }->setEnabled(enabled);
    }
    bool operator==(const WebCore::AudioTrackPrivate& track) const { return track == m_trackPrivate.get(); }

private:
    RemoteAudioTrackProxy(GPUConnectionToWebProcess&, WebCore::AudioTrackPrivate&, WebCore::MediaPlayerIdentifier);

    // AudioTrackPrivateClient
    void enabledChanged(bool) final;
    void configurationChanged(const WebCore::PlatformAudioTrackConfiguration&) final;

    // TrackPrivateBaseClient
    void idChanged(WebCore::TrackID) final;
    void labelChanged(const AtomString&) final;
    void languageChanged(const AtomString&) final;
    void willRemove() final;

    AudioTrackPrivateRemoteConfiguration configuration();
    void configurationChanged();

    ThreadSafeWeakPtr<GPUConnectionToWebProcess> m_connectionToWebProcess;
    Ref<WebCore::AudioTrackPrivate> m_trackPrivate;
    WebCore::TrackID m_id;
    WebCore::MediaPlayerIdentifier m_mediaPlayerIdentifier;
    bool m_enabled { false };
    size_t m_clientId { 0 };
};

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS) && ENABLE(VIDEO)
