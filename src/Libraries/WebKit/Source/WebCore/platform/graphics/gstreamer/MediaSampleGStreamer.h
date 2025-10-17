/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 16, 2023.
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

#if ENABLE(VIDEO) && USE(GSTREAMER)

#include "FloatSize.h"
#include "GStreamerCommon.h"
#include "MediaSample.h"
#include "VideoFrameTimeMetadata.h"

namespace WebCore {

class MediaSampleGStreamer : public MediaSample {
public:
    static Ref<MediaSampleGStreamer> create(GRefPtr<GstSample>&& sample, const FloatSize& presentationSize, TrackID id)
    {
        return adoptRef(*new MediaSampleGStreamer(WTFMove(sample), presentationSize, id));
    }

    static Ref<MediaSampleGStreamer> createFakeSample(GstCaps*, const MediaTime& pts, const MediaTime& dts, const MediaTime& duration, const FloatSize& presentationSize, TrackID);

    void extendToTheBeginning();
    MediaTime presentationTime() const override { return m_pts; }
    MediaTime decodeTime() const override { return m_dts; }
    MediaTime duration() const override { return m_duration; }
    TrackID trackID() const override { return m_trackId; }
    size_t sizeInBytes() const override { return m_size; }
    FloatSize presentationSize() const override { return m_presentationSize; }
    void offsetTimestampsBy(const MediaTime&) override;
    void setTimestamps(const MediaTime&, const MediaTime&) override;
    Ref<MediaSample> createNonDisplayingCopy() const override;
    SampleFlags flags() const override { return m_flags; }
    PlatformSample platformSample() const override;
    PlatformSample::Type platformSampleType() const override { return PlatformSample::GStreamerSampleType; }
    void dump(PrintStream&) const override;

    const GRefPtr<GstSample>& sample() const { return m_sample; }

protected:
    MediaSampleGStreamer(GRefPtr<GstSample>&&, const FloatSize& presentationSize, TrackID);
    virtual ~MediaSampleGStreamer() = default;

private:
    MediaSampleGStreamer(const FloatSize& presentationSize, TrackID);

    MediaTime m_pts;
    MediaTime m_dts;
    MediaTime m_duration;
    TrackID m_trackId;
    size_t m_size { 0 };
    GRefPtr<GstSample> m_sample;
    FloatSize m_presentationSize;
    MediaSample::SampleFlags m_flags { MediaSample::IsSync };
};

} // namespace WebCore.

#endif // ENABLE(VIDEO) && USE(GSTREAMER)
