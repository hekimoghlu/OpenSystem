/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 20, 2023.
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

#if ENABLE(MEDIA_SOURCE)

#include "VideoTrackPrivate.h"
#include <webm/dom_types.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

struct VideoInfo;

class VideoTrackPrivateWebM final : public VideoTrackPrivate {
    WTF_MAKE_TZONE_ALLOCATED(VideoTrackPrivateWebM);
public:
    static Ref<VideoTrackPrivateWebM> create(webm::TrackEntry&&);
    virtual ~VideoTrackPrivateWebM() = default;

    TrackID id() const final;
    AtomString label() const final;
    AtomString language() const final;
    int trackIndex() const final;
    std::optional<bool> defaultEnabled() const final;
    uint32_t width() const;
    uint32_t height() const;

private:
    VideoTrackPrivateWebM(webm::TrackEntry&&);

    void setFormatDescription(Ref<VideoInfo>&&);

    String codec() const;
    double framerate() const;
    PlatformVideoColorSpace colorSpace() const;
    void updateConfiguration();

    webm::TrackEntry m_track;
    mutable AtomString m_label;
    mutable AtomString m_language;
    RefPtr<VideoInfo> m_formatDescription;
};

}

#endif // ENABLE(MEDIA_SOURCE)
