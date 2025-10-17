/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 28, 2023.
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

#if ENABLE(WEB_AUDIO)

#include "AudioDestination.h"
#include "PushPullFIFO.h"
#include <wtf/Lock.h>
#include <wtf/RefPtr.h>

namespace WebCore {

class AudioBus;
class MultiChannelResampler;

// Base class for audio destinations that may need resampling.
// The subclass should use m_outputBus to access the rendering.
class AudioDestinationResampler : public AudioDestination {
public:
    WEBCORE_EXPORT AudioDestinationResampler(AudioIOCallback&, unsigned numberOfOutputChannels, float inputSampleRate, float outputSampleRate);
    WEBCORE_EXPORT virtual ~AudioDestinationResampler();

    WEBCORE_EXPORT void start(Function<void(Function<void()>&&)>&& dispatchToRenderThread, CompletionHandler<void(bool)>&&) final;
    WEBCORE_EXPORT void stop(CompletionHandler<void(bool)>&&) final;

protected:
    WEBCORE_EXPORT void setIsPlaying(bool);
    bool isPlaying() final { return m_isPlaying; }
    WEBCORE_EXPORT unsigned framesPerBuffer() const final;

    // The caller is expected to call both pullRendered and render.
    // The caller fills the m_outputBus channel data pointers before calling this.
    // Returns the number of frames to render.
    WEBCORE_EXPORT size_t pullRendered(size_t numberOfFrames);
    WEBCORE_EXPORT bool render(double sampleTime, MonotonicTime hostTime, size_t framesToRender);

private:
    void renderOnRenderingThreadIfPlaying(size_t framesToRender);

    virtual void startRendering(CompletionHandler<void(bool)>&&) = 0;
    virtual void stopRendering(CompletionHandler<void(bool)>&&) = 0;

protected:
    // To pass the data from FIFO to the audio device callback.
    Ref<AudioBus> m_outputBus;

private:
    // To push the rendered result from WebAudio graph into the FIFO.
    Ref<AudioBus> m_renderBus;

    // Resolves the buffer size mismatch between the WebAudio engine and
    // the callback function from the actual audio device.
    Lock m_fifoLock;
    PushPullFIFO m_fifo WTF_GUARDED_BY_LOCK(m_fifoLock);

    std::unique_ptr<MultiChannelResampler> m_resampler;
    AudioIOPosition m_outputTimestamp;

    Lock m_dispatchToRenderThreadLock;
    Function<void(Function<void()>&&)> m_dispatchToRenderThread WTF_GUARDED_BY_LOCK(m_dispatchToRenderThreadLock);

    std::atomic<bool> m_isPlaying;
};

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
