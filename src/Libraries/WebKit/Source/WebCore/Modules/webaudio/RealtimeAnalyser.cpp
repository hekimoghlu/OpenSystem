/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 9, 2023.
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

#include "RealtimeAnalyser.h"

#include "AudioBus.h"
#include "AudioNode.h"
#include "AudioUtilities.h"
#include "FFTFrame.h"
#include "VectorMath.h"
#include <JavaScriptCore/Float32Array.h>
#include <JavaScriptCore/Uint8Array.h>
#include <algorithm>
#include <complex>
#include <wtf/MainThread.h>
#include <wtf/MathExtras.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RealtimeAnalyser);

RealtimeAnalyser::RealtimeAnalyser(OptionSet<NoiseInjectionPolicy> policies)
    : m_inputBuffer(InputBufferSize)
    , m_downmixBus(AudioBus::create(1, AudioUtilities::renderQuantumSize))
    , m_noiseInjectionPolicies(policies)
{
    m_analysisFrame = makeUnique<FFTFrame>(DefaultFFTSize);
}

RealtimeAnalyser::~RealtimeAnalyser() = default;

bool RealtimeAnalyser::setFftSize(size_t size)
{
    ASSERT(isMainThread());

    // Only allow powers of two.
    unsigned log2size = static_cast<unsigned>(log2(size));
    bool isPOT(1UL << log2size == size);

    if (!isPOT || size > MaxFFTSize || size < MinFFTSize)
        return false;

    if (m_fftSize != size) {
        m_analysisFrame = makeUnique<FFTFrame>(size);
        // m_magnitudeBuffer has size = fftSize / 2 because it contains floats reduced from complex values in m_analysisFrame.
        m_magnitudeBuffer.resize(size / 2);
        m_fftSize = size;
    }

    return true;
}

void RealtimeAnalyser::writeInput(AudioBus* bus, size_t framesToProcess)
{
    bool isBusGood = bus && bus->numberOfChannels() > 0 && bus->channel(0)->length() >= framesToProcess;
    ASSERT(isBusGood);
    if (!isBusGood)
        return;
        
    // FIXME : allow to work with non-FFTSize divisible chunking
    bool isDestinationGood = m_writeIndex < m_inputBuffer.size() && m_writeIndex + framesToProcess <= m_inputBuffer.size();
    ASSERT(isDestinationGood);
    if (!isDestinationGood)
        return;    
    
    // Perform real-time analysis
    auto destination = m_inputBuffer.span().subspan(m_writeIndex);

    // Clear the bus and downmix the input according to the down mixing rules.
    // Then save the result in the m_inputBuffer at the appropriate place.
    m_downmixBus->zero();
    m_downmixBus->sumFrom(*bus);
    memcpySpan(destination, m_downmixBus->channel(0)->span().first(framesToProcess));

    m_writeIndex += framesToProcess;
    if (m_writeIndex >= InputBufferSize)
        m_writeIndex = 0;

    // A new render quantum has been processed so we should do the FFT analysis again.
    m_shouldDoFFTAnalysis = true;
}

namespace {

void applyWindow(std::span<float> p, size_t n)
{
    ASSERT(isMainThread());
    
    // Blackman window
    double alpha = 0.16;
    double a0 = 0.5 * (1 - alpha);
    double a1 = 0.5;
    double a2 = 0.5 * alpha;
    
    for (unsigned i = 0; i < n; ++i) {
        double x = static_cast<double>(i) / static_cast<double>(n);
        double window = a0 - a1 * cos(2 * piDouble * x) + a2 * cos(4 * piDouble * x);
        p[i] *= float(window);
    }
}

} // namespace

void RealtimeAnalyser::doFFTAnalysisIfNecessary()
{    
    ASSERT(isMainThread());

    if (!m_shouldDoFFTAnalysis)
        return;

    m_shouldDoFFTAnalysis = false;

    // Unroll the input buffer into a temporary buffer, where we'll apply an analysis window followed by an FFT.
    size_t fftSize = this->fftSize();
    
    AudioFloatArray temporaryBuffer(fftSize);
    auto inputBuffer = m_inputBuffer.span();
    auto tempP = temporaryBuffer.span();

    // Take the previous fftSize values from the input buffer and copy into the temporary buffer.
    unsigned writeIndex = m_writeIndex;
    if (writeIndex < fftSize) {
        memcpySpan(tempP, inputBuffer.subspan(writeIndex - fftSize + InputBufferSize, fftSize - writeIndex));
        memcpySpan(tempP.subspan(fftSize - writeIndex), inputBuffer.first(writeIndex));
    } else 
        memcpySpan(tempP, inputBuffer.subspan(writeIndex - fftSize, fftSize));

    
    // Window the input samples.
    applyWindow(tempP, fftSize);
    
    // Do the analysis.
    m_analysisFrame->doFFT(tempP);

    auto& realP = m_analysisFrame->realData();
    auto& imagP = m_analysisFrame->imagData();

    // Blow away the packed nyquist component.
    imagP[0] = 0;
    
    // Normalize so than an input sine wave at 0dBfs registers as 0dBfs (undo FFT scaling factor).
    const double magnitudeScale = 1.0 / fftSize;

    // A value of 0 does no averaging with the previous result.  Larger values produce slower, but smoother changes.
    double k = m_smoothingTimeConstant;
    k = std::max(0.0, k);
    k = std::min(1.0, k);    
    
    // Convert the analysis data from complex to magnitude and average with the previous result.
    auto destination = magnitudeBuffer().span();
    for (size_t i = 0; i < destination.size(); ++i) {
        std::complex<double> c(realP[i], imagP[i]);
        double scalarMagnitude = std::abs(c) * magnitudeScale;        
        destination[i] = static_cast<float>(k * destination[i] + (1 - k) * scalarMagnitude);
    }

    if (m_noiseInjectionPolicies)
        AudioUtilities::applyNoise(destination, 0.25);
}

void RealtimeAnalyser::getFloatFrequencyData(Float32Array& destinationArray)
{
    ASSERT(isMainThread());
        
    doFFTAnalysisIfNecessary();
    
    // Convert from linear magnitude to floating-point decibels.
    size_t length = std::min<size_t>(magnitudeBuffer().size(), destinationArray.length());
    VectorMath::linearToDecibels(magnitudeBuffer().span().first(length), destinationArray.typedMutableSpan());
}

void RealtimeAnalyser::getByteFrequencyData(Uint8Array& destinationArray)
{
    ASSERT(isMainThread());
        
    doFFTAnalysisIfNecessary();
    
    // Convert from linear magnitude to unsigned-byte decibels.
    size_t sourceLength = magnitudeBuffer().size();
    size_t destinationLength = destinationArray.length();
    size_t length = std::min(sourceLength, destinationLength);
    if (length > 0) {
        const double rangeScaleFactor = m_maxDecibels == m_minDecibels ? 1 : 1 / (m_maxDecibels - m_minDecibels);
        const double minDecibels = m_minDecibels;

        auto source = magnitudeBuffer().span();
        auto destination = destinationArray.mutableSpan();
        
        for (size_t i = 0; i < length; ++i) {
            float linearValue = source[i];
            double dbMag = !linearValue ? minDecibels : AudioUtilities::linearToDecibels(linearValue);
            
            // The range m_minDecibels to m_maxDecibels will be scaled to byte values from 0 to UCHAR_MAX.
            double scaledValue = UCHAR_MAX * (dbMag - minDecibels) * rangeScaleFactor;

            // Clip to valid range.
            if (scaledValue < 0)
                scaledValue = 0;
            if (scaledValue > UCHAR_MAX)
                scaledValue = UCHAR_MAX;
            
            destination[i] = static_cast<unsigned char>(scaledValue);
        }
    }
}

void RealtimeAnalyser::getFloatTimeDomainData(Float32Array& destinationArray)
{
    ASSERT(isMainThread());
    
    size_t destinationLength = destinationArray.length();
    size_t fftSize = this->fftSize();
    size_t length = std::min(fftSize, destinationLength);
    if (length > 0) {
        bool isInputBufferGood = m_inputBuffer.size() == InputBufferSize && m_inputBuffer.size() > fftSize;
        ASSERT(isInputBufferGood);
        if (!isInputBufferGood)
            return;
        
        auto inputBuffer = m_inputBuffer.span();
        auto destination = destinationArray.typedMutableSpan();
        
        unsigned writeIndex = m_writeIndex;
        
        for (size_t i = 0; i < length; ++i) {
            // Buffer access is protected due to modulo operation.
            destination[i] = inputBuffer[(i + writeIndex - fftSize + InputBufferSize) % InputBufferSize];
        }
    }
}

void RealtimeAnalyser::getByteTimeDomainData(Uint8Array& destinationArray)
{
    ASSERT(isMainThread());

    size_t destinationLength = destinationArray.length();
    size_t fftSize = this->fftSize();
    size_t length = std::min(fftSize, destinationLength);
    if (length > 0) {
        bool isInputBufferGood = m_inputBuffer.size() == InputBufferSize && m_inputBuffer.size() > fftSize;
        ASSERT(isInputBufferGood);
        if (!isInputBufferGood)
            return;

        auto inputBuffer = m_inputBuffer.span();
        auto destination = destinationArray.mutableSpan();
        
        unsigned writeIndex = m_writeIndex;

        for (size_t i = 0; i < length; ++i) {
            // Buffer access is protected due to modulo operation.
            float value = inputBuffer[(i + writeIndex - fftSize + InputBufferSize) % InputBufferSize];

            // Scale from nominal -1 -> +1 to unsigned byte.
            double scaledValue = 128 * (value + 1);

            // Clip to valid range.
            if (scaledValue < 0)
                scaledValue = 0;
            if (scaledValue > UCHAR_MAX)
                scaledValue = UCHAR_MAX;
            
            destination[i] = static_cast<unsigned char>(scaledValue);
        }
    }
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
