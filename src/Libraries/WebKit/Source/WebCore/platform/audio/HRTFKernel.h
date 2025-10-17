/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 17, 2023.
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
#ifndef HRTFKernel_h
#define HRTFKernel_h

#include "FFTFrame.h"
#include <memory>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>
#include <wtf/Vector.h>

namespace WebCore {

class AudioChannel;
    
// HRTF stands for Head-Related Transfer Function.
// HRTFKernel is a frequency-domain representation of an impulse-response used as part of the spatialized panning system.
// For a given azimuth / elevation angle there will be one HRTFKernel for the left ear transfer function, and one for the right ear.
// The leading delay (average group delay) for each impulse response is extracted:
//      m_fftFrame is the frequency-domain representation of the impulse response with the delay removed
//      m_frameDelay is the leading delay of the original impulse response.
class HRTFKernel : public RefCounted<HRTFKernel> {
public:
    // Note: this is destructive on the passed in AudioChannel.
    // The length of channel must be a power of two.
    static Ref<HRTFKernel> create(AudioChannel* channel, size_t fftSize, float sampleRate)
    {
        return adoptRef(*new HRTFKernel(channel, fftSize, sampleRate));
    }

    static Ref<HRTFKernel> create(std::unique_ptr<FFTFrame> fftFrame, float frameDelay, float sampleRate)
    {
        return adoptRef(*new HRTFKernel(WTFMove(fftFrame), frameDelay, sampleRate));
    }

    // Given two HRTFKernels, and an interpolation factor x: 0 -> 1, returns an interpolated HRTFKernel.
    static RefPtr<HRTFKernel> createInterpolatedKernel(HRTFKernel* kernel1, HRTFKernel* kernel2, float x);
  
    FFTFrame* fftFrame() { return m_fftFrame.get(); }
    
    size_t fftSize() const;
    float frameDelay() const { return m_frameDelay; }

    float sampleRate() const { return m_sampleRate; }
    double nyquist() const { return 0.5 * sampleRate(); }

    // Converts back into impulse-response form.
    std::unique_ptr<AudioChannel> createImpulseResponse();

private:
    // Note: this is destructive on the passed in AudioChannel.
    HRTFKernel(AudioChannel*, size_t fftSize, float sampleRate);
    
    HRTFKernel(std::unique_ptr<FFTFrame> fftFrame, float frameDelay, float sampleRate)
        : m_fftFrame(WTFMove(fftFrame))
        , m_frameDelay(frameDelay)
        , m_sampleRate(sampleRate)
    {
    }
    
    std::unique_ptr<FFTFrame> m_fftFrame;
    float m_frameDelay { 0 };
    float m_sampleRate;
};

typedef Vector<RefPtr<HRTFKernel>> HRTFKernelList;

} // namespace WebCore

#endif // HRTFKernel_h
