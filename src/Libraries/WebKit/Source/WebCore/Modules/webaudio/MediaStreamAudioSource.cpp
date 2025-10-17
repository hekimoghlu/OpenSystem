/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 19, 2023.
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
#include "MediaStreamAudioSource.h"

#if ENABLE(MEDIA_STREAM)

#include "NotImplemented.h"
#include "PlatformAudioData.h"

namespace WebCore {

MediaStreamAudioSource::MediaStreamAudioSource(float sampleRate)
    : RealtimeMediaSource(CaptureDevice { { }, CaptureDevice::DeviceType::Microphone, "MediaStreamAudioDestinationNode"_s })
{
    m_currentSettings.setSampleRate(sampleRate);

#if USE(GSTREAMER)
    gst_audio_info_init(&m_info);
#endif
}

MediaStreamAudioSource::~MediaStreamAudioSource() = default;

const RealtimeMediaSourceCapabilities& MediaStreamAudioSource::capabilities()
{
    // FIXME: implement this.
    // https://bugs.webkit.org/show_bug.cgi?id=122430
    notImplemented();
    return RealtimeMediaSourceCapabilities::emptyCapabilities();
}

const RealtimeMediaSourceSettings& MediaStreamAudioSource::settings()
{
    // FIXME: implement this.
    // https://bugs.webkit.org/show_bug.cgi?id=122430
    notImplemented();
    return m_currentSettings;
}

#if !PLATFORM(COCOA) && !USE(GSTREAMER)
void MediaStreamAudioSource::consumeAudio(AudioBus&, size_t)
{
}
#endif

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM)
