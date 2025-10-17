/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 24, 2023.
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

#if ENABLE(MEDIA_STREAM)

#include "RealtimeMediaSource.h"
#include <wtf/Lock.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

#if USE(GSTREAMER)
#include "GStreamerCommon.h"
#endif

namespace WebCore {

class AudioBus;
class PlatformAudioData;
class RealtimeMediaSourceCapabilities;

class MediaStreamAudioSource final : public RealtimeMediaSource, public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<MediaStreamAudioSource, WTF::DestructionThread::MainRunLoop> {
public:
    static Ref<MediaStreamAudioSource> create(float sampleRate) { return adoptRef(*new MediaStreamAudioSource { sampleRate }); }

    ~MediaStreamAudioSource();
    WTF_ABSTRACT_THREAD_SAFE_REF_COUNTED_AND_CAN_MAKE_WEAK_PTR_IMPL;

    const RealtimeMediaSourceCapabilities& capabilities() final;
    const RealtimeMediaSourceSettings& settings() final;

    const String& deviceId() const { return m_deviceId; }
    void setDeviceId(const String& deviceId) { m_deviceId = deviceId; }

    void consumeAudio(AudioBus&, size_t numberOfFrames);

private:
    explicit MediaStreamAudioSource(float sampleRate);

    bool isCaptureSource() const final { return false; }

    String m_deviceId;
    RealtimeMediaSourceSettings m_currentSettings;
    std::unique_ptr<PlatformAudioData> m_audioBuffer;
#if USE(AVFOUNDATION) || USE(GSTREAMER)
    size_t m_numberOfFrames { 0 };
#endif
#if USE(GSTREAMER)
    GstAudioInfo m_info;
    GRefPtr<GstCaps> m_caps;
#endif
};

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM)
