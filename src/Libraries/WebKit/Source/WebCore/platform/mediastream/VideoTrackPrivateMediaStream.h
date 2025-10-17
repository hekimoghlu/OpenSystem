/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 30, 2025.
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

#if ENABLE(MEDIA_STREAM)

#include "MediaStreamTrackPrivate.h"
#include "VideoTrackPrivate.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/StringToIntegerConversion.h>

namespace WebCore {

class VideoTrackPrivateMediaStream final : public VideoTrackPrivate {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(VideoTrackPrivateMediaStream);
    WTF_MAKE_NONCOPYABLE(VideoTrackPrivateMediaStream)
public:
    static Ref<VideoTrackPrivateMediaStream> create(MediaStreamTrackPrivate& streamTrack)
    {
        return adoptRef(*new VideoTrackPrivateMediaStream(streamTrack));
    }

    void setTrackIndex(int index) { m_index = index; }

    MediaStreamTrackPrivate& streamTrack() { return m_streamTrack.get(); }

private:
    VideoTrackPrivateMediaStream(MediaStreamTrackPrivate& track)
        : m_streamTrack(track)
    {
    }

    Kind kind() const final { return Kind::Main; }
    std::optional<AtomString> trackUID() const { return AtomString { m_streamTrack->id() }; }
    AtomString label() const final { return AtomString { m_streamTrack->label() }; }
    AtomString language() const final { return emptyAtom(); }
    int trackIndex() const final { return m_index; }

    Ref<MediaStreamTrackPrivate> m_streamTrack;
    int m_index { 0 };
};

}

#endif // ENABLE(MEDIA_STREAM)
