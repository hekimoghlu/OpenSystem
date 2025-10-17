/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 16, 2022.
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

#include <WebCore/AudioTrackPrivate.h>
#include <WebCore/MediaPlayerIdentifier.h>
#include <wtf/TZoneMalloc.h>

namespace WebKit {

class GPUProcessConnection;
class MediaPlayerPrivateRemote;
struct AudioTrackPrivateRemoteConfiguration;

class AudioTrackPrivateRemote final : public WebCore::AudioTrackPrivate {
    WTF_MAKE_TZONE_ALLOCATED(AudioTrackPrivateRemote);
    WTF_MAKE_NONCOPYABLE(AudioTrackPrivateRemote)
public:
    static Ref<AudioTrackPrivateRemote> create(GPUProcessConnection& gpuProcessConnection, WebCore::MediaPlayerIdentifier playerIdentifier, AudioTrackPrivateRemoteConfiguration&& configuration)
    {
        return adoptRef(*new AudioTrackPrivateRemote(gpuProcessConnection, playerIdentifier, WTFMove(configuration)));
    }

    WebCore::TrackID id() const final { return m_id; }
    void updateConfiguration(AudioTrackPrivateRemoteConfiguration&&);

private:
    AudioTrackPrivateRemote(GPUProcessConnection&, WebCore::MediaPlayerIdentifier, AudioTrackPrivateRemoteConfiguration&&);

    using AudioTrackKind = WebCore::AudioTrackPrivate::Kind;
    AudioTrackKind kind() const final { return m_kind; }
    AtomString label() const final { return AtomString { m_label }; }
    AtomString language() const final { return AtomString { m_language }; }
    int trackIndex() const final { return m_trackIndex; }
    void setEnabled(bool) final;
    MediaTime startTimeVariance() const final { return m_startTimeVariance; }

    ThreadSafeWeakPtr<GPUProcessConnection> m_gpuProcessConnection;
    AudioTrackKind m_kind { AudioTrackKind::None };
    WebCore::TrackID m_id;
    String m_label;
    String m_language;
    int m_trackIndex { -1 };

    MediaTime m_startTimeVariance { MediaTime::zeroTime() };
    WebCore::MediaPlayerIdentifier m_playerIdentifier;
};

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS) && ENABLE(VIDEO)
