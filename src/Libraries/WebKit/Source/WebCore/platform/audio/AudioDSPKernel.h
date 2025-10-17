/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 6, 2022.
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
#ifndef AudioDSPKernel_h
#define AudioDSPKernel_h

#include "AudioDSPKernelProcessor.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

// AudioDSPKernel does the processing for one channel of an AudioDSPKernelProcessor.

class AudioDSPKernel {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(AudioDSPKernel);
    WTF_MAKE_NONCOPYABLE(AudioDSPKernel);
public:
    AudioDSPKernel(AudioDSPKernelProcessor* kernelProcessor)
        : m_kernelProcessor(kernelProcessor)
        , m_sampleRate(kernelProcessor->sampleRate())
    {
    }

    AudioDSPKernel(float sampleRate)
        : m_kernelProcessor(0)
        , m_sampleRate(sampleRate)
    {
    }

    virtual ~AudioDSPKernel() { };

    // Subclasses must override process() to do the processing and reset() to reset DSP state.
    virtual void process(std::span<const float> source, std::span<float> destination) = 0;

    // Subclasses that have AudioParams must override this to process the AudioParams.
    virtual void processOnlyAudioParams(size_t) { }

    virtual void reset() = 0;

    float sampleRate() const { return m_sampleRate; }
    double nyquist() const { return 0.5 * sampleRate(); }

    AudioDSPKernelProcessor* processor() { return m_kernelProcessor; }
    const AudioDSPKernelProcessor* processor() const { return m_kernelProcessor; }

    virtual double tailTime() const = 0;
    virtual double latencyTime() const = 0;
    virtual bool requiresTailProcessing() const = 0;

protected:
    AudioDSPKernelProcessor* m_kernelProcessor;
    float m_sampleRate;
};

} // namespace WebCore

#endif // AudioDSPKernel_h
