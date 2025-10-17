/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 31, 2024.
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

#include <WebCore/MediaPlayerIdentifier.h>
#include <WebCore/VideoTrackPrivate.h>
#include <wtf/TZoneMalloc.h>

namespace WebKit {

class GPUProcessConnection;
class MediaPlayerPrivateRemote;
struct VideoTrackPrivateRemoteConfiguration;

class VideoTrackPrivateRemote
    : public WebCore::VideoTrackPrivate {
    WTF_MAKE_TZONE_ALLOCATED(VideoTrackPrivateRemote);
    WTF_MAKE_NONCOPYABLE(VideoTrackPrivateRemote)
public:
    static Ref<VideoTrackPrivateRemote> create(GPUProcessConnection& gpuProcessConnection, WebCore::MediaPlayerIdentifier playerIdentifier, VideoTrackPrivateRemoteConfiguration&& configuration)
    {
        return adoptRef(*new VideoTrackPrivateRemote(gpuProcessConnection, playerIdentifier, WTFMove(configuration)));
    }

    void updateConfiguration(VideoTrackPrivateRemoteConfiguration&&);

    using VideoTrackKind = WebCore::VideoTrackPrivate::Kind;
    VideoTrackKind kind() const final { return m_kind; }
    WebCore::TrackID id() const final { return m_id; }
    AtomString label() const final { return AtomString { m_label }; }
    AtomString language() const final { return AtomString { m_language }; }
    int trackIndex() const final { return m_trackIndex; }
    MediaTime startTimeVariance() const final { return m_startTimeVariance; }

private:
    VideoTrackPrivateRemote(GPUProcessConnection&, WebCore::MediaPlayerIdentifier, VideoTrackPrivateRemoteConfiguration&&);

    void setSelected(bool) final;

    ThreadSafeWeakPtr<GPUProcessConnection> m_gpuProcessConnection;
    VideoTrackKind m_kind { VideoTrackKind::None };
    String m_label;
    String m_language;
    int m_trackIndex { -1 };
    MediaTime m_startTimeVariance { MediaTime::zeroTime() };
    WebCore::TrackID m_id;
    WebCore::MediaPlayerIdentifier m_playerIdentifier;
};

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS) && ENABLE(VIDEO)
