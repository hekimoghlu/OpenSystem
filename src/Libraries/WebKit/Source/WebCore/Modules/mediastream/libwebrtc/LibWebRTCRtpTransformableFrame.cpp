/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 28, 2023.
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
#include "LibWebRTCRtpTransformableFrame.h"
#include <wtf/TZoneMallocInlines.h>

#if ENABLE(WEB_RTC) && USE(LIBWEBRTC)

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN

#include <webrtc/api/frame_transformer_interface.h>

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(LibWebRTCRtpTransformableFrame);

LibWebRTCRtpTransformableFrame::LibWebRTCRtpTransformableFrame(std::unique_ptr<webrtc::TransformableFrameInterface>&& frame, bool isAudioSenderFrame)
    : m_rtcFrame(WTFMove(frame))
    , m_isAudioSenderFrame(isAudioSenderFrame)
{
}

LibWebRTCRtpTransformableFrame::~LibWebRTCRtpTransformableFrame()
{
}

std::unique_ptr<webrtc::TransformableFrameInterface> LibWebRTCRtpTransformableFrame::takeRTCFrame()
{
    return WTFMove(m_rtcFrame);
}

std::span<const uint8_t> LibWebRTCRtpTransformableFrame::data() const
{
    if (!m_rtcFrame)
        return { };
    auto data = m_rtcFrame->GetData();
    return unsafeMakeSpan(data.begin(), data.size());
}

void LibWebRTCRtpTransformableFrame::setData(std::span<const uint8_t> data)
{
    if (m_rtcFrame)
        m_rtcFrame->SetData({ data.data(), data.size() });
}

bool LibWebRTCRtpTransformableFrame::isKeyFrame() const
{
    ASSERT(m_rtcFrame);
    auto* videoFrame = static_cast<webrtc::TransformableVideoFrameInterface*>(m_rtcFrame.get());
    return videoFrame && videoFrame->IsKeyFrame();
}

uint64_t LibWebRTCRtpTransformableFrame::timestamp() const
{
    return m_rtcFrame ? m_rtcFrame->GetTimestamp() : 0;
}

RTCEncodedAudioFrameMetadata LibWebRTCRtpTransformableFrame::audioMetadata() const
{
    if (!m_rtcFrame)
        return { };

    Vector<uint32_t> cssrcs;
    if (!m_isAudioSenderFrame) {
        auto* audioFrame = static_cast<webrtc::TransformableAudioFrameInterface*>(m_rtcFrame.get());
        auto contributingSources = audioFrame->GetContributingSources();
        cssrcs = Vector<uint32_t>(contributingSources.size(), [&](size_t cptr) {
            return contributingSources[cptr];
        });
    }
    return { m_rtcFrame->GetSsrc(), WTFMove(cssrcs) };
}

RTCEncodedVideoFrameMetadata LibWebRTCRtpTransformableFrame::videoMetadata() const
{
    if (!m_rtcFrame)
        return { };
    auto* videoFrame = static_cast<webrtc::TransformableVideoFrameInterface*>(m_rtcFrame.get());
    auto metadata = videoFrame->Metadata();

    std::optional<int64_t> frameId;
    if (metadata.GetFrameId())
        frameId = *metadata.GetFrameId();

    Vector<int64_t> dependencies;
    for (auto value : metadata.GetFrameDependencies())
        dependencies.append(value);

    return { frameId, WTFMove(dependencies), metadata.GetWidth(), metadata.GetHeight(), metadata.GetSpatialIndex(), metadata.GetTemporalIndex(), m_rtcFrame->GetSsrc() };
}

} // namespace WebCore

#endif // ENABLE(WEB_RTC) && USE(LIBWEBRTC)
