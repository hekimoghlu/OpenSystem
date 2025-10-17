/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 27, 2022.
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
#ifndef FFTFrame_h
#define FFTFrame_h

#include "AudioArray.h"

#if USE(GSTREAMER)
#include "GUniquePtrGStreamer.h"
#endif // USE(GSTREAMER)

#if USE(ACCELERATE)
#include <Accelerate/Accelerate.h>
#endif

#include <memory>
#include <wtf/Forward.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/UniqueArray.h>

namespace WebCore {

// Defines the interface for an "FFT frame", an object which is able to perform a forward
// and reverse FFT, internally storing the resultant frequency-domain data.

class FFTFrame {
    WTF_MAKE_TZONE_ALLOCATED(FFTFrame);
public:
    // The constructors, destructor, and methods up to the CROSS-PLATFORM section have platform-dependent implementations.

    FFTFrame(unsigned fftSize);
    FFTFrame(); // creates a blank/empty frame for later use with createInterpolatedFrame()
    FFTFrame(const FFTFrame& frame);
    ~FFTFrame();

    static void initialize();
    void doFFT(std::span<const float> data);
    void doInverseFFT(std::span<float> data);
    void multiply(const FFTFrame& frame); // multiplies ourself with frame : effectively operator*=()
    void scaleFFT(float factor);

    AudioFloatArray& realData() { return m_realData; }
    AudioFloatArray& imagData() { return m_imagData; }
    const AudioFloatArray& realData() const { return m_realData; }
    const AudioFloatArray& imagData() const { return m_imagData; }

    static int minFFTSize();
    static int maxFFTSize();

    void print(); // for debugging

    // CROSS-PLATFORM
    // The remaining public methods have cross-platform implementations:

    // Interpolates from frame1 -> frame2 as x goes from 0.0 -> 1.0
    static std::unique_ptr<FFTFrame> createInterpolatedFrame(const FFTFrame& frame1, const FFTFrame& frame2, double x);

    void doPaddedFFT(std::span<const float> data); // zero-padding with data.size() <= fftSize
    double extractAverageGroupDelay();
    void addConstantGroupDelay(double sampleFrameDelay);

    unsigned fftSize() const { return m_FFTSize; }
    unsigned log2FFTSize() const { return m_log2FFTSize; }

private:
    unsigned m_FFTSize;
    unsigned m_log2FFTSize;

    void interpolateFrequencyComponents(const FFTFrame& frame1, const FFTFrame& frame2, double x);

#if USE(ACCELERATE)
    DSPSplitComplex& dspSplitComplex() { return m_frame; }
    DSPSplitComplex dspSplitComplex() const { return m_frame; }

    static FFTSetup fftSetupForSize(unsigned fftSize);

    FFTSetup m_FFTSetup;

    DSPSplitComplex m_frame;
#endif

#if USE(GSTREAMER)
    GUniquePtr<GstFFTF32> m_fft;
    GUniquePtr<GstFFTF32> m_inverseFft;
    UniqueArray<GstFFTF32Complex> m_complexData;
#endif // USE(GSTREAMER)

    AudioFloatArray m_realData;
    AudioFloatArray m_imagData;
};

} // namespace WebCore

#endif // FFTFrame_h
