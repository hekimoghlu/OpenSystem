/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 27, 2024.
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

#if USE(LIBWEBRTC)
#include "LibWebRTCAudioModule.h"
#endif

#include <wtf/Function.h>
#include <wtf/LoggerHelper.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeWeakPtr.h>

namespace WTF {
class MediaTime;
}

namespace WebCore {

class AudioStreamDescription;
class PlatformAudioData;

class WEBCORE_EXPORT AudioMediaStreamTrackRenderer : public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<AudioMediaStreamTrackRenderer, WTF::DestructionThread::Main>, public LoggerHelper {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(AudioMediaStreamTrackRenderer, WEBCORE_EXPORT);
public:
    struct Init {
        Function<void()>&& crashCallback;
#if USE(LIBWEBRTC)
        RefPtr<LibWebRTCAudioModule> audioModule;
#endif
#if !RELEASE_LOG_DISABLED
        const Logger& logger;
        uint64_t logIdentifier;
#endif
    };
    static RefPtr<AudioMediaStreamTrackRenderer> create(Init&&);
    virtual ~AudioMediaStreamTrackRenderer() = default;

    static String defaultDeviceID();

    virtual void start(CompletionHandler<void()>&&) = 0;
    virtual void stop() = 0;
    virtual void clear() = 0;
    // May be called on a background thread. It should only be called after start/before stop is called.
    virtual void pushSamples(const WTF::MediaTime&, const PlatformAudioData&, const AudioStreamDescription&, size_t) = 0;

    virtual void setVolume(float);
    float volume() const;

    virtual void setAudioOutputDevice(const String&);

protected:
    explicit AudioMediaStreamTrackRenderer(Init&&);

#if !RELEASE_LOG_DISABLED
    const Logger& logger() const final;
    uint64_t logIdentifier() const final;

    ASCIILiteral logClassName() const final;
    WTFLogChannel& logChannel() const final;
#endif

#if USE(LIBWEBRTC)
    LibWebRTCAudioModule* audioModule();
#endif

    void crashed();

private:
    // Main thread writable members
    float m_volume { 1 };
    Function<void()> m_crashCallback;

#if USE(LIBWEBRTC)
    RefPtr<LibWebRTCAudioModule> m_audioModule;
#endif

#if !RELEASE_LOG_DISABLED
    Ref<const Logger> m_logger;
    const uint64_t m_logIdentifier;
#endif
};

inline void AudioMediaStreamTrackRenderer::setVolume(float volume)
{
    m_volume = volume;
}

inline float AudioMediaStreamTrackRenderer::volume() const
{
    return m_volume;
}

inline void AudioMediaStreamTrackRenderer::crashed()
{
    if (m_crashCallback)
        m_crashCallback();
}

#if USE(LIBWEBRTC)
inline LibWebRTCAudioModule* AudioMediaStreamTrackRenderer::audioModule()
{
    return m_audioModule.get();
}
#endif

inline void AudioMediaStreamTrackRenderer::setAudioOutputDevice(const String&)
{
}

}

#endif // ENABLE(MEDIA_STREAM)
