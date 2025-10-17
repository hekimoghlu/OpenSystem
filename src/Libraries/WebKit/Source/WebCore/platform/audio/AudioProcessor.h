/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 9, 2024.
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
#ifndef AudioProcessor_h
#define AudioProcessor_h

#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

class AudioBus;

// AudioProcessor is an abstract base class representing an audio signal processing object with a single input and a single output,
// where the number of input channels equals the number of output channels.  It can be used as one part of a complex DSP algorithm,
// or as the processor for a basic (one input - one output) AudioNode.

class AudioProcessor {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(AudioProcessor);
    WTF_MAKE_NONCOPYABLE(AudioProcessor);
public:
    AudioProcessor(float sampleRate, unsigned numberOfChannels)
        : m_initialized(false)
        , m_numberOfChannels(numberOfChannels)
        , m_sampleRate(sampleRate)
    {
    }

    virtual ~AudioProcessor() = default;

    // Full initialization can be done here instead of in the constructor.
    virtual void initialize() = 0;
    virtual void uninitialize() = 0;

    // Processes the source to destination bus.  The number of channels must match in source and destination.
    virtual void process(const AudioBus* source, AudioBus* destination, size_t framesToProcess) = 0;

    // Forces all AudioParams in the processor to run the timeline, bypassing any other processing the processor
    // would do in process().
    virtual void processOnlyAudioParams(size_t) { }

    // Resets filter state
    virtual void reset() = 0;

    virtual void setNumberOfChannels(unsigned) = 0;
    virtual unsigned numberOfChannels() const = 0;

    bool isInitialized() const { return m_initialized; }

    float sampleRate() const { return m_sampleRate; }

    virtual double tailTime() const = 0;
    virtual double latencyTime() const = 0;
    virtual bool requiresTailProcessing() const = 0;

    enum class Type : uint8_t { Biquad, Delay, IIR, WaveShaper };
    virtual Type processorType() const = 0;

protected:
    bool m_initialized;
    unsigned m_numberOfChannels;
    float m_sampleRate;
};

} // namespace WebCore

#endif // AudioProcessor_h
