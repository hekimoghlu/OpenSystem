/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 18, 2025.
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

#if USE(GSTREAMER_WEBRTC)

#include "GRefPtrGStreamer.h"
#include "GUniquePtrGStreamer.h"
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class GStreamerRTPPacketizer : public ThreadSafeRefCounted<GStreamerRTPPacketizer> {
    WTF_MAKE_NONCOPYABLE(GStreamerRTPPacketizer);
    WTF_MAKE_FAST_ALLOCATED;
public:
    explicit GStreamerRTPPacketizer(GRefPtr<GstElement>&& encoder, GRefPtr<GstElement>&& payloader, GUniquePtr<GstStructure>&& encodingParameters, std::optional<int>&&);
    virtual ~GStreamerRTPPacketizer();

    GstElement* bin() const { return m_bin.get(); }
    GstElement* payloader() const { return m_payloader.get(); }

    WARN_UNUSED_RETURN GUniquePtr<GstStructure> rtpParameters() const;

    void configureExtensions();
    void ensureMidExtension(const String&);

    String rtpStreamId() const;
    std::optional<int> payloadType() const;
    unsigned currentSequenceNumberOffset() const;
    void setSequenceNumberOffset(unsigned);

    std::optional<std::pair<unsigned, GstStructure*>> stats() const;
    void startUpdatingStats();
    void stopUpdatingStats();

    virtual void updateStats() { };

    void reconfigure(GUniquePtr<GstStructure>&&);

protected:
    int findLastExtensionId(const GstCaps*);

    GRefPtr<GstElement> m_bin;
    GRefPtr<GstElement> m_inputQueue;
    GRefPtr<GstElement> m_outputQueue;
    GRefPtr<GstElement> m_encoder;
    GRefPtr<GstElement> m_payloader;
    GRefPtr<GstElement> m_capsFilter;
    GRefPtr<GstElement> m_valve;

    GUniquePtr<GstStructure> m_encodingParameters;
    GUniquePtr<GstStructure> m_stats;

private:
    void setPayloadType(int);
    void updateStatsFromRTPExtensions();
    void applyEncodingParameters(const GstStructure*) const;
    virtual void configure(const GstStructure*) const { };

    GRefPtr<GstRTPHeaderExtension> m_midExtension;
    GRefPtr<GstRTPHeaderExtension> m_ridExtension;

    unsigned m_lastExtensionId { 0 };

    unsigned long m_statsPadProbeId { 0 };

    String m_mid;
    String m_rid;
    std::optional<int> m_payloadType;
};

} // namespace WebCore

#endif // USE(GSTREAMER_WEBRTC)
