/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 17, 2024.
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
#ifndef AudioDestination_h
#define AudioDestination_h

#include "AudioBus.h"
#include "AudioIOCallback.h"
#include <memory>
#include <wtf/AbstractRefCounted.h>
#include <wtf/CompletionHandler.h>
#include <wtf/Lock.h>
#include <wtf/MediaTime.h>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

// AudioDestination is an abstraction for audio hardware I/O.
// The audio hardware periodically calls the AudioIOCallback render() method asking it to render/output the next render quantum of audio.
// It optionally will pass in local/live audio input when it calls render().

class AudioDestination : public AbstractRefCounted {
public:
    // Pass in (numberOfInputChannels > 0) if live/local audio input is desired.
    // Port-specific device identification information for live/local input streams can be passed in the inputDeviceId.
    WEBCORE_EXPORT static Ref<AudioDestination> create(AudioIOCallback&, const String& inputDeviceId, unsigned numberOfInputChannels, unsigned numberOfOutputChannels, float sampleRate);

    virtual ~AudioDestination() = default;

    void clearCallback();

    virtual void start(Function<void(Function<void()>&&)>&& dispatchToRenderThread, CompletionHandler<void(bool)>&& = [](bool) { }) = 0;
    virtual void stop(CompletionHandler<void(bool)>&& = [](bool) { }) = 0;
    virtual bool isPlaying() = 0;

    // Sample-rate conversion may happen in AudioDestination to the hardware sample-rate
    virtual float sampleRate() const { return m_sampleRate; }
    WEBCORE_EXPORT static float hardwareSampleRate();

    virtual unsigned framesPerBuffer() const = 0;
    virtual WTF::MediaTime outputLatency() const { return MediaTime::zeroTime(); }

    // maxChannelCount() returns the total number of output channels of the audio hardware.
    // A value of 0 indicates that the number of channels cannot be configured and
    // that only stereo (2-channel) destinations can be created.
    // The numberOfOutputChannels parameter of AudioDestination::create() is allowed to
    // be a value: 1 <= numberOfOutputChannels <= maxChannelCount(),
    // or if maxChannelCount() equals 0, then numberOfOutputChannels must be 2.
    static unsigned long maxChannelCount();

    void callRenderCallback(AudioBus* sourceBus, AudioBus* destinationBus, size_t framesToProcess, const AudioIOPosition& outputPosition);

protected:
    explicit AudioDestination(AudioIOCallback&, float sampleRate);

    Lock m_callbackLock;
    AudioIOCallback* m_callback WTF_GUARDED_BY_LOCK(m_callbackLock) { nullptr };

private:
    const float m_sampleRate;
};

inline AudioDestination::AudioDestination(AudioIOCallback& callback, float sampleRate)
    : m_sampleRate(sampleRate)
{
    Locker locker { m_callbackLock };
    m_callback = &callback;
}

inline void AudioDestination::clearCallback()
{
    Locker locker { m_callbackLock };
    m_callback = nullptr;
}

inline void AudioDestination::callRenderCallback(AudioBus* sourceBus, AudioBus* destinationBus, size_t framesToProcess, const AudioIOPosition& outputPosition)
{
    if (m_callbackLock.tryLock()) {
        Locker locker { AdoptLock, m_callbackLock };
        if (m_callback) {
            m_callback->render(sourceBus, destinationBus, framesToProcess, outputPosition);
            return;
        }
    }
    destinationBus->zero();
}

} // namespace WebCore

#endif // AudioDestination_h
