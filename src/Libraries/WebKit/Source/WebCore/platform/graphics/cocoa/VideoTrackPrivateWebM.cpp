/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 19, 2023.
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
#include "config.h"
#include "VideoTrackPrivateWebM.h"
#include <wtf/TZoneMallocInlines.h>

#if ENABLE(MEDIA_SOURCE)

#include "MediaSample.h"

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(VideoTrackPrivateWebM);

Ref<VideoTrackPrivateWebM> VideoTrackPrivateWebM::create(webm::TrackEntry&& trackEntry)
{
    return adoptRef(*new VideoTrackPrivateWebM(WTFMove(trackEntry)));
}

VideoTrackPrivateWebM::VideoTrackPrivateWebM(webm::TrackEntry&& trackEntry)
    : m_track(WTFMove(trackEntry))
{
    if (m_track.is_enabled.is_present())
        setSelected(m_track.is_enabled.value());

    updateConfiguration();
}

void VideoTrackPrivateWebM::setFormatDescription(Ref<VideoInfo>&& formatDescription)
{
    if (m_formatDescription && *m_formatDescription == formatDescription)
        return;
    m_formatDescription = WTFMove(formatDescription);
    updateConfiguration();
}

TrackID VideoTrackPrivateWebM::id() const
{
    if (m_track.track_uid.is_present())
        return m_track.track_uid.value();
    if (m_track.track_number.is_present())
        return m_track.track_number.value();
    ASSERT_NOT_REACHED();
    return 0;
}

std::optional<bool> VideoTrackPrivateWebM::defaultEnabled() const
{
    if (m_track.is_enabled.is_present())
        return m_track.is_enabled.value();
    return std::nullopt;
}

AtomString VideoTrackPrivateWebM::label() const
{
    if (m_label.isNull())
        m_label = m_track.name.is_present() ? AtomString::fromUTF8(m_track.name.value()) : emptyAtom();
    return m_label;
}

AtomString VideoTrackPrivateWebM::language() const
{
    if (m_language.isNull())
        m_language = m_track.language.is_present() ? AtomString::fromUTF8(m_track.language.value()) : emptyAtom();
    return m_language;
}

int VideoTrackPrivateWebM::trackIndex() const
{
    if (m_track.track_number.is_present())
        return m_track.track_number.value();
    return 0;
}

String VideoTrackPrivateWebM::codec() const
{
    if (m_formatDescription) {
        if (!m_formatDescription->codecString.isEmpty())
            return m_formatDescription->codecString;
        return String::fromLatin1(m_formatDescription->codecName.string().data());
    }

    if (!m_track.codec_id.is_present())
        return emptyString();

    StringView codecID { std::span { m_track.codec_id.value() } };

    if (codecID == "V_VP9"_s)
        return "vp09"_s;

    if (codecID == "V_VP8"_s)
        return "vp08"_s;

    return emptyString();
}

uint32_t VideoTrackPrivateWebM::width() const
{
    if (m_formatDescription)
        return m_formatDescription->size.width();

    if (!m_track.video.is_present())
        return 0;

    auto& video = m_track.video.value();
    if (video.display_width.is_present())
        return video.display_width.value();

    if (video.pixel_width.is_present())
        return video.pixel_width.value();

    return 0;
}

uint32_t VideoTrackPrivateWebM::height() const
{
    if (m_formatDescription)
        return m_formatDescription->size.height();

    if (!m_track.video.is_present())
        return 0;

    auto& video = m_track.video.value();
    if (video.display_height.is_present())
        return video.display_height.value();

    if (video.pixel_height.is_present())
        return video.pixel_height.value();

    return 0;
}

double VideoTrackPrivateWebM::framerate() const
{
    if (!m_track.video.is_present())
        return 0;

    auto& video = m_track.video.value();
    if (video.frame_rate.is_present())
        return video.frame_rate.value();

    if (m_track.default_duration.is_present()) {
        static constexpr double nanosecondsPerSecond = 1000 * 1000 * 1000;
        return nanosecondsPerSecond / m_track.default_duration.value();
    }

    return 0;
}

PlatformVideoColorSpace VideoTrackPrivateWebM::colorSpace() const
{
    if (m_formatDescription)
        return m_formatDescription->colorSpace;
    return { };
}

void VideoTrackPrivateWebM::updateConfiguration()
{
IGNORE_WARNINGS_BEGIN("c99-designator")
    PlatformVideoTrackConfiguration configuration {
        { .codec = codec() },
        .width = width(),
        .height = height(),
        .colorSpace = colorSpace(),
        .framerate = framerate(),
        .spatialVideoMetadata = { }
    };
IGNORE_WARNINGS_END
    setConfiguration(WTFMove(configuration));
}

}

#endif // ENABLE(MEDIA_SOURCE)
