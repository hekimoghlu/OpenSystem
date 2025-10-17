/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 29, 2022.
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

#if ENABLE(VIDEO)

#include "TrackBase.h"
#include "VideoTrackPrivateClient.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakHashSet.h>

namespace WebCore {

class MediaDescription;
class VideoTrack;
class VideoTrackClient;
class VideoTrackConfiguration;
class VideoTrackList;
class VideoTrackPrivate;

class VideoTrack final : public MediaTrackBase, private VideoTrackPrivateClient {
    WTF_MAKE_TZONE_ALLOCATED(VideoTrack);
public:
    static Ref<VideoTrack> create(ScriptExecutionContext* context, VideoTrackPrivate& trackPrivate)
    {
        return adoptRef(*new VideoTrack(context, trackPrivate));
    }
    virtual ~VideoTrack();

    static const AtomString& signKeyword();

    bool selected() const { return m_selected; }
    virtual void setSelected(const bool);

    void addClient(VideoTrackClient&);
    void clearClient(VideoTrackClient&);

    size_t inbandTrackIndex();

    void setKind(const AtomString&) final;
    void setLanguage(const AtomString&) final;

    const MediaDescription& description() const;

    VideoTrackConfiguration& configuration() const { return m_configuration; }

    void setPrivate(VideoTrackPrivate&);
#if !RELEASE_LOG_DISABLED
    void setLogger(const Logger&, uint64_t) final;
#endif

private:
    VideoTrack(ScriptExecutionContext*, VideoTrackPrivate&);

    bool isValidKind(const AtomString&) const final;

    // VideoTrackPrivateClient
    void selectedChanged(bool) final;
    void configurationChanged(const PlatformVideoTrackConfiguration&) final;

    // TrackPrivateBaseClient
    void idChanged(TrackID) final;
    void labelChanged(const AtomString&) final;
    void languageChanged(const AtomString&) final;
    void willRemove() final;

    bool enabled() const final { return selected(); }

    void updateKindFromPrivate();
    void updateConfigurationFromPrivate();

#if !RELEASE_LOG_DISABLED
    ASCIILiteral logClassName() const final { return "VideoTrack"_s; }
#endif

    WeakPtr<VideoTrackList> m_videoTrackList;
    WeakHashSet<VideoTrackClient> m_clients;
    Ref<VideoTrackPrivate> m_private;
    Ref<VideoTrackConfiguration> m_configuration;
    bool m_selected { false };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::VideoTrack)
    static bool isType(const WebCore::TrackBase& track) { return track.type() == WebCore::TrackBase::VideoTrack; }
SPECIALIZE_TYPE_TRAITS_END()

#endif
