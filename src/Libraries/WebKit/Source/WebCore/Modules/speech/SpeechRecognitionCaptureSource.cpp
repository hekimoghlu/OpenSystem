/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 27, 2023.
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
#include "SpeechRecognitionCaptureSource.h"
#include <wtf/TZoneMallocInlines.h>

#if ENABLE(MEDIA_STREAM)
#include "CaptureDeviceManager.h"
#include "RealtimeMediaSourceCenter.h"
#include "SpeechRecognitionUpdate.h"
#endif

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(SpeechRecognitionCaptureSource);

void SpeechRecognitionCaptureSource::mute()
{
#if ENABLE(MEDIA_STREAM)
    m_impl->mute();
#endif
}

#if ENABLE(MEDIA_STREAM)

std::optional<CaptureDevice> SpeechRecognitionCaptureSource::findCaptureDevice()
{
    std::optional<CaptureDevice> captureDevice;
    auto devices = RealtimeMediaSourceCenter::singleton().audioCaptureFactory().audioCaptureDeviceManager().captureDevices();
    for (auto device : devices) {
        if (!device.enabled())
            continue;

        if (!captureDevice)
            captureDevice = device;

        if (device.isDefault()) {
            captureDevice = device;
            break;
        }
    }
    return captureDevice;
}

CaptureSourceOrError SpeechRecognitionCaptureSource::createRealtimeMediaSource(const CaptureDevice& captureDevice, PageIdentifier pageIdentifier)
{
    return RealtimeMediaSourceCenter::singleton().audioCaptureFactory().createAudioCaptureSource(captureDevice, { "SpeechID"_s, "SpeechID"_s }, { }, pageIdentifier);
}

SpeechRecognitionCaptureSource::SpeechRecognitionCaptureSource(SpeechRecognitionConnectionClientIdentifier clientIdentifier, DataCallback&& dataCallback, StateUpdateCallback&& stateUpdateCallback, Ref<RealtimeMediaSource>&& source)
    : m_impl(makeUnique<SpeechRecognitionCaptureSourceImpl>(clientIdentifier, WTFMove(dataCallback), WTFMove(stateUpdateCallback), WTFMove(source)))
{
}

#endif

} // namespace WebCore
