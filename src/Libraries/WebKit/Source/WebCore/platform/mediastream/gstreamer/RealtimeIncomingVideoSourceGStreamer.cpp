/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 3, 2024.
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

#if USE(GSTREAMER_WEBRTC)
#include "RealtimeIncomingVideoSourceGStreamer.h"

#include "GStreamerCommon.h"
#include "GStreamerWebRTCUtils.h"
#include "VideoFrameGStreamer.h"
#include <wtf/text/MakeString.h>

GST_DEBUG_CATEGORY(webkit_webrtc_incoming_video_debug);
#define GST_CAT_DEFAULT webkit_webrtc_incoming_video_debug

namespace WebCore {

RealtimeIncomingVideoSourceGStreamer::RealtimeIncomingVideoSourceGStreamer(AtomString&& videoTrackId)
    : RealtimeIncomingSourceGStreamer(CaptureDevice { WTFMove(videoTrackId), CaptureDevice::DeviceType::Camera, emptyString() })
{
    static std::once_flag debugRegisteredFlag;
    std::call_once(debugRegisteredFlag, [] {
        GST_DEBUG_CATEGORY_INIT(webkit_webrtc_incoming_video_debug, "webkitwebrtcincomingvideo", 0, "WebKit WebRTC incoming video");
    });
}

const RealtimeMediaSourceSettings& RealtimeIncomingVideoSourceGStreamer::settings()
{
    if (m_currentSettings)
        return m_currentSettings.value();

    RealtimeMediaSourceSettings settings;
    RealtimeMediaSourceSupportedConstraints constraints;

    auto& size = this->size();
    if (!size.isZero()) {
        constraints.setSupportsWidth(true);
        constraints.setSupportsHeight(true);
        settings.setWidth(size.width());
        settings.setHeight(size.height());
    }

    if (double frameRate = this->frameRate()) {
        constraints.setSupportsFrameRate(true);
        settings.setFrameRate(frameRate);
    }

    settings.setSupportedConstraints(constraints);

    m_currentSettings = WTFMove(settings);
    return m_currentSettings.value();
}

void RealtimeIncomingVideoSourceGStreamer::settingsDidChange(OptionSet<RealtimeMediaSourceSettings::Flag> settings)
{
    if (settings.containsAny({ RealtimeMediaSourceSettings::Flag::Width, RealtimeMediaSourceSettings::Flag::Height, RealtimeMediaSourceSettings::Flag::FrameRate }))
        m_currentSettings = std::nullopt;
}

void RealtimeIncomingVideoSourceGStreamer::ensureSizeAndFramerate(const GRefPtr<GstCaps>& caps)
{
    if (auto size = getVideoResolutionFromCaps(caps.get()))
        setIntrinsicSize({ static_cast<int>(size->width()), static_cast<int>(size->height()) });

    int frameRateNumerator, frameRateDenominator;
    auto* structure = gst_caps_get_structure(caps.get(), 0);
    if (!gst_structure_get_fraction(structure, "framerate", &frameRateNumerator, &frameRateDenominator))
        return;

    double framerate;
    gst_util_fraction_to_double(frameRateNumerator, frameRateDenominator, &framerate);
    setFrameRate(framerate);
}

void RealtimeIncomingVideoSourceGStreamer::dispatchSample(GRefPtr<GstSample>&& sample)
{
    ASSERT(isMainThread());
    auto* buffer = gst_sample_get_buffer(sample.get());
    auto* caps = gst_sample_get_caps(sample.get());
    if (!caps) {
        GST_WARNING_OBJECT(bin(), "Received sample without caps, bailing out.");
        return;
    }

    ensureSizeAndFramerate(GRefPtr<GstCaps>(caps));

    videoFrameAvailable(VideoFrameGStreamer::create(WTFMove(sample), intrinsicSize(), fromGstClockTime(GST_BUFFER_PTS(buffer))), { });
}

const GstStructure* RealtimeIncomingVideoSourceGStreamer::stats()
{
    m_stats.reset(gst_structure_new_empty("incoming-video-stats"));

    forEachVideoFrameObserver([&](auto& observer) {
        auto stats = observer.queryAdditionalStats();
        if (!stats)
            return;

        gstStructureForeach(stats.get(), [&](auto id, auto value) -> bool {
            gstStructureIdSetValue(m_stats.get(), id, value);
            return TRUE;
        });
    });
    return m_stats.get();
}

#undef GST_CAT_DEFAULT

} // namespace WebCore

#endif // USE(GSTREAMER_WEBRTC)
