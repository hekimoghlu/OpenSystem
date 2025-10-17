/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 17, 2022.
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
#include <WebCore/InbandTextTrackPrivate.h>
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
struct TextTrackPrivateRemoteConfiguration;

class RemoteTextTrackProxy final
    : public ThreadSafeRefCounted<RemoteTextTrackProxy, WTF::DestructionThread::Main>
    , private WebCore::InbandTextTrackPrivateClient {
    WTF_MAKE_TZONE_ALLOCATED(RemoteTextTrackProxy);
public:
    static Ref<RemoteTextTrackProxy> create(GPUConnectionToWebProcess& connectionToWebProcess, WebCore::InbandTextTrackPrivate& trackPrivate, WebCore::MediaPlayerIdentifier mediaPlayerIdentifier)
    {
        return adoptRef(*new RemoteTextTrackProxy(connectionToWebProcess, trackPrivate, mediaPlayerIdentifier));
    }

    virtual ~RemoteTextTrackProxy();

    WebCore::TrackID id() const { return m_trackPrivate->id(); }
    void setMode(WebCore::InbandTextTrackPrivate::Mode mode) { m_trackPrivate->setMode(mode); }
    bool operator==(const WebCore::InbandTextTrackPrivate& track) const { return track == m_trackPrivate.get(); }

private:
    RemoteTextTrackProxy(GPUConnectionToWebProcess&, WebCore::InbandTextTrackPrivate&, WebCore::MediaPlayerIdentifier);

    // InbandTextTrackPrivateClient
    virtual void addDataCue(const MediaTime& start, const MediaTime& end, std::span<const uint8_t>);

#if ENABLE(DATACUE_VALUE)
    virtual void addDataCue(const MediaTime& start, const MediaTime& end, Ref<WebCore::SerializedPlatformDataCue>&&, const String&);
    virtual void updateDataCue(const MediaTime& start, const MediaTime& end, WebCore::SerializedPlatformDataCue&);
    virtual void removeDataCue(const MediaTime& start, const MediaTime& end, WebCore::SerializedPlatformDataCue&);
#endif

    virtual void addGenericCue(WebCore::InbandGenericCue&);
    virtual void updateGenericCue(WebCore::InbandGenericCue&);
    virtual void removeGenericCue(WebCore::InbandGenericCue&);

    virtual void parseWebVTTFileHeader(String&&);
    virtual void parseWebVTTCueData(std::span<const uint8_t>);
    virtual void parseWebVTTCueData(WebCore::ISOWebVTTCue&&);

    // TrackPrivateBaseClient
    void idChanged(WebCore::TrackID) final;
    void labelChanged(const AtomString&) final;
    void languageChanged(const AtomString&) final;
    void willRemove() final;

    TextTrackPrivateRemoteConfiguration& configuration();
    void configurationChanged();

    ThreadSafeWeakPtr<GPUConnectionToWebProcess> m_connectionToWebProcess;
    Ref<WebCore::InbandTextTrackPrivate> m_trackPrivate;
    WebCore::TrackID m_id;
    WebCore::MediaPlayerIdentifier m_mediaPlayerIdentifier;
    size_t m_clientId { 0 };
};

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS) && ENABLE(VIDEO)
