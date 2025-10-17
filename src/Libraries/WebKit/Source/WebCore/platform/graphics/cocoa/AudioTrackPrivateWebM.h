/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 12, 2024.
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

#include "AudioTrackPrivate.h"
#include <webm/dom_types.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

struct AudioInfo;

class AudioTrackPrivateWebM final : public AudioTrackPrivate {
    WTF_MAKE_TZONE_ALLOCATED(AudioTrackPrivateWebM);
public:
    static Ref<AudioTrackPrivateWebM> create(webm::TrackEntry&&);
    virtual ~AudioTrackPrivateWebM() = default;

    TrackID id() const final;
    AtomString label() const final;
    AtomString language() const final;
    int trackIndex() const final;
    std::optional<bool> defaultEnabled() const final;
    std::optional<MediaTime> codecDelay() const;
    void setDiscardPadding(const MediaTime&);
    std::optional<MediaTime> discardPadding() const;

private:
    AudioTrackPrivateWebM(webm::TrackEntry&&);

    String codec() const;
    uint32_t sampleRate() const;
    uint32_t numberOfChannels() const;
    void setFormatDescription(Ref<AudioInfo>&&) final;
    void updateConfiguration();

    webm::TrackEntry m_track;
    RefPtr<AudioInfo> m_formatDescription;
    MediaTime m_discardPadding { MediaTime::invalidTime() };
    mutable AtomString m_label;
    mutable AtomString m_language;
};

}

#endif // ENABLE(MEDIA_SOURCE)
