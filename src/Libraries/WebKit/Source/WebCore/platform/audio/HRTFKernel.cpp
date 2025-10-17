/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 13, 2022.
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

#if ENABLE(WEB_AUDIO)

#include "HRTFKernel.h"

#include "AudioChannel.h"
#include "Biquad.h"
#include "FFTFrame.h"
#include "FloatConversion.h"
#include <wtf/MathExtras.h>

namespace WebCore {

// Takes the input AudioChannel as an input impulse response and calculates the average group delay.
// This represents the initial delay before the most energetic part of the impulse response.
// The sample-frame delay is removed from the impulseP impulse response, and this value  is returned.
// the length of the passed in AudioChannel must be a power of 2.
static float extractAverageGroupDelay(AudioChannel* channel, size_t analysisFFTSize)
{
    ASSERT(channel);
        
    auto impulseP = channel->mutableSpan();
    
    bool isSizeGood = channel->length() >= analysisFFTSize;
    ASSERT(isSizeGood);
    if (!isSizeGood)
        return 0;
    
    // Check for power-of-2.
    ASSERT(1UL << static_cast<unsigned>(log2(analysisFFTSize)) == analysisFFTSize);

    FFTFrame estimationFrame(analysisFFTSize);
    estimationFrame.doFFT(impulseP);

    float frameDelay = narrowPrecisionToFloat(estimationFrame.extractAverageGroupDelay());
    estimationFrame.doInverseFFT(impulseP);

    return frameDelay;
}

HRTFKernel::HRTFKernel(AudioChannel* channel, size_t fftSize, float sampleRate)
    : m_sampleRate(sampleRate)
{
    ASSERT(channel);

    // Determine the leading delay (average group delay) for the response.
    m_frameDelay = extractAverageGroupDelay(channel, fftSize / 2);

    auto impulseResponse = channel->mutableSpan();
    size_t responseLength = channel->length();

    // We need to truncate to fit into 1/2 the FFT size (with zero padding) in order to do proper convolution.
    size_t truncatedResponseLength = std::min(responseLength, fftSize / 2); // truncate if necessary to max impulse response length allowed by FFT

    // Quick fade-out (apply window) at truncation point
    unsigned numberOfFadeOutFrames = static_cast<unsigned>(sampleRate / 4410); // 10 sample-frames @44.1KHz sample-rate
    ASSERT(numberOfFadeOutFrames < truncatedResponseLength);
    if (numberOfFadeOutFrames < truncatedResponseLength) {
        for (unsigned i = truncatedResponseLength - numberOfFadeOutFrames; i < truncatedResponseLength; ++i) {
            float x = 1.0f - static_cast<float>(i - (truncatedResponseLength - numberOfFadeOutFrames)) / numberOfFadeOutFrames;
            impulseResponse[i] *= x;
        }
    }

    m_fftFrame = makeUnique<FFTFrame>(fftSize);
    m_fftFrame->doPaddedFFT(impulseResponse.first(truncatedResponseLength));
}

size_t HRTFKernel::fftSize() const
{
    return m_fftFrame->fftSize();
}

std::unique_ptr<AudioChannel> HRTFKernel::createImpulseResponse()
{
    auto channel = makeUnique<AudioChannel>(fftSize());
    FFTFrame fftFrame(*m_fftFrame);

    // Add leading delay back in.
    fftFrame.addConstantGroupDelay(m_frameDelay);
    fftFrame.doInverseFFT(channel->mutableSpan());

    return channel;
}

// Interpolates two kernels with x: 0 -> 1 and returns the result.
RefPtr<HRTFKernel> HRTFKernel::createInterpolatedKernel(HRTFKernel* kernel1, HRTFKernel* kernel2, float x)
{
    ASSERT(kernel1 && kernel2);
    if (!kernel1 || !kernel2)
        return nullptr;
 
    ASSERT(x >= 0.0 && x < 1.0);
    x = std::min(1.0f, std::max(0.0f, x));
    
    float sampleRate1 = kernel1->sampleRate();
    float sampleRate2 = kernel2->sampleRate();
    ASSERT(sampleRate1 == sampleRate2);
    if (sampleRate1 != sampleRate2)
        return nullptr;
    
    float frameDelay = (1 - x) * kernel1->frameDelay() + x * kernel2->frameDelay();
    
    std::unique_ptr<FFTFrame> interpolatedFrame = FFTFrame::createInterpolatedFrame(*kernel1->fftFrame(), *kernel2->fftFrame(), x);
    return HRTFKernel::create(WTFMove(interpolatedFrame), frameDelay, sampleRate1);
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
