/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 15, 2025.
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
#include "RealtimeIncomingAudioSource.h"

#if USE(LIBWEBRTC)

#include "LibWebRTCAudioFormat.h"
#include "LibWebRTCAudioModule.h"
#include "Logging.h"

namespace WebCore {

RealtimeIncomingAudioSource::RealtimeIncomingAudioSource(rtc::scoped_refptr<webrtc::AudioTrackInterface>&& audioTrack, String&& audioTrackId)
    : RealtimeMediaSource(CaptureDevice { WTFMove(audioTrackId), CaptureDevice::DeviceType::Microphone, "remote audio"_s })
    , m_audioTrack(WTFMove(audioTrack))
{
    ASSERT(m_audioTrack);
    m_audioTrack->RegisterObserver(this);
}

RealtimeIncomingAudioSource::~RealtimeIncomingAudioSource()
{
    stop();
    m_audioTrack->UnregisterObserver(this);
}

void RealtimeIncomingAudioSource::startProducingData()
{
    m_audioTrack->AddSink(this);
}

void RealtimeIncomingAudioSource::stopProducingData()
{
    m_audioTrack->RemoveSink(this);
}

void RealtimeIncomingAudioSource::OnChanged()
{
    callOnMainThread([protectedThis = Ref { *this }] {
        if (protectedThis->m_audioTrack->state() == webrtc::MediaStreamTrackInterface::kEnded)
            protectedThis->end();
    });
}

const RealtimeMediaSourceCapabilities& RealtimeIncomingAudioSource::capabilities()
{
    return RealtimeMediaSourceCapabilities::emptyCapabilities();
}

const RealtimeMediaSourceSettings& RealtimeIncomingAudioSource::settings()
{
    return m_currentSettings;
}

void RealtimeIncomingAudioSource::setAudioModule(RefPtr<LibWebRTCAudioModule>&& audioModule)
{
    ASSERT(!m_audioModule);
    m_audioModule = WTFMove(audioModule);
}

}

#endif // USE(LIBWEBRTC)
