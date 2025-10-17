/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 17, 2025.
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

#if ENABLE(VIDEO) && USE(GSTREAMER)

#include "InbandTextTrackPrivateGStreamer.h"

#include <wtf/Lock.h>
#include <wtf/text/StringToIntegerConversion.h>

namespace WebCore {

GST_DEBUG_CATEGORY(webkit_text_track_debug);
#define GST_CAT_DEFAULT webkit_text_track_debug

static void ensureTextTrackDebugCategoryInitialized()
{
    static std::once_flag debugRegisteredFlag;
    std::call_once(debugRegisteredFlag, [] {
        GST_DEBUG_CATEGORY_INIT(webkit_text_track_debug, "webkittexttrack", 0, "WebKit Text Track");
    });
}

InbandTextTrackPrivateGStreamer::InbandTextTrackPrivateGStreamer(unsigned index, GRefPtr<GstPad>&& pad, bool shouldHandleStreamStartEvent)
    : InbandTextTrackPrivate(CueFormat::WebVTT)
    , TrackPrivateBaseGStreamer(TrackPrivateBaseGStreamer::TrackType::Text, this, index, WTFMove(pad), shouldHandleStreamStartEvent)
    , m_kind(Kind::Subtitles)
{
    ensureTextTrackDebugCategoryInitialized();
    installUpdateConfigurationHandlers();
}

InbandTextTrackPrivateGStreamer::InbandTextTrackPrivateGStreamer(unsigned index, GRefPtr<GstPad>&& pad, TrackID trackId)
    : InbandTextTrackPrivate(CueFormat::WebVTT)
    , TrackPrivateBaseGStreamer(TrackPrivateBaseGStreamer::TrackType::Text, this, index, WTFMove(pad), trackId)
    , m_kind(Kind::Subtitles)
{
    ensureTextTrackDebugCategoryInitialized();
    installUpdateConfigurationHandlers();
}

InbandTextTrackPrivateGStreamer::InbandTextTrackPrivateGStreamer(unsigned index, GstStream* stream)
    : InbandTextTrackPrivate(CueFormat::WebVTT)
    , TrackPrivateBaseGStreamer(TrackPrivateBaseGStreamer::TrackType::Text, this, index, stream)
{
    ensureTextTrackDebugCategoryInitialized();
    installUpdateConfigurationHandlers();

    GST_INFO("Track %d got stream start for stream %" PRIu64 ". GStreamer stream-id: %s", m_index, m_id, m_gstStreamId.string().utf8().data());

    GST_DEBUG("Stream %" GST_PTR_FORMAT, m_stream.get());
    auto caps = adoptGRef(gst_stream_get_caps(m_stream.get()));
    m_kind = doCapsHaveType(caps.get(), "closedcaption/"_s) ? Kind::Captions : Kind::Subtitles;
}

void InbandTextTrackPrivateGStreamer::tagsChanged(GRefPtr<GstTagList>&& tags)
{
    ASSERT(isMainThread());
    if (!tags)
        return;

    if (!updateTrackIDFromTags(tags))
        return;

    GST_DEBUG_OBJECT(objectForLogging(), "Text track ID set from container-specific-track-id tag %" G_GUINT64_FORMAT, *m_trackID);
    notifyClients([trackID = *m_trackID](auto& client) {
        client.idChanged(trackID);
    });
}

void InbandTextTrackPrivateGStreamer::handleSample(GRefPtr<GstSample>&& sample)
{
    {
        Locker locker { m_sampleMutex };
        m_pendingSamples.append(WTFMove(sample));
    }

    RefPtr<InbandTextTrackPrivateGStreamer> protectedThis(this);
    m_notifier->notify(MainThreadNotification::NewSample, [protectedThis] {
        protectedThis->notifyTrackOfSample();
    });
}

void InbandTextTrackPrivateGStreamer::notifyTrackOfSample()
{
    Vector<GRefPtr<GstSample>> samples;
    {
        Locker locker { m_sampleMutex };
        m_pendingSamples.swap(samples);
    }

    for (auto& sample : samples) {
        GstBuffer* buffer = gst_sample_get_buffer(sample.get());
        if (!buffer) {
            GST_WARNING("Track %d got sample with no buffer.", m_index);
            continue;
        }
        GstMappedBuffer mappedBuffer(buffer, GST_MAP_READ);
        ASSERT(mappedBuffer);
        if (!mappedBuffer) {
            GST_WARNING("Track %d unable to map buffer.", m_index);
            continue;
        }

        GST_INFO("Track %d parsing sample: %.*s", m_index, static_cast<int>(mappedBuffer.size()),
            reinterpret_cast<char*>(mappedBuffer.data()));
        ASSERT(isMainThread());
        ASSERT(!hasClients() || hasOneClient());
        notifyMainThreadClient([&](auto& client) {
            downcast<InbandTextTrackPrivateClient>(client).parseWebVTTCueData(mappedBuffer.span<uint8_t>());
        });
    }
}

#undef GST_CAT_DEFAULT

} // namespace WebCore

#endif // ENABLE(VIDEO) && USE(GSTREAMER)
