/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 17, 2021.
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
// Mac OS X - specific FFTFrame implementation

#include "config.h"

#if ENABLE(WEB_AUDIO)

#if OS(DARWIN)

#include "FFTFrame.h"

#include "VectorMath.h"
#include <wtf/Lock.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/Vector.h>

namespace WebCore {

constexpr unsigned kMinFFTPow2Size = 2;
constexpr unsigned kMaxFFTPow2Size = 24;

static Lock fftSetupsLock;

static Vector<FFTSetup>& fftSetups() WTF_REQUIRES_LOCK(fftSetupsLock)
{
    ASSERT(fftSetupsLock.isHeld());
    static NeverDestroyed<Vector<FFTSetup>> fftSetups(kMaxFFTPow2Size, nullptr);
    return fftSetups;
}

// Normal constructor: allocates for a given fftSize
FFTFrame::FFTFrame(unsigned fftSize)
    : m_realData(fftSize)
    , m_imagData(fftSize)
{
    m_FFTSize = fftSize;
    m_log2FFTSize = static_cast<unsigned>(log2(fftSize));

    // We only allow power of two
    ASSERT(1UL << m_log2FFTSize == m_FFTSize);

    // Lazily create and share fftSetup with other frames
    m_FFTSetup = fftSetupForSize(fftSize);

    // Setup frame data
    m_frame.realp = m_realData.data();
    m_frame.imagp = m_imagData.data();
}

// Creates a blank/empty frame (interpolate() must later be called)
FFTFrame::FFTFrame()
{
    // Later will be set to correct values when interpolate() is called
    m_frame.realp = 0;
    m_frame.imagp = 0;

    m_FFTSize = 0;
    m_log2FFTSize = 0;
}

// Copy constructor
FFTFrame::FFTFrame(const FFTFrame& frame)
    : m_FFTSize(frame.m_FFTSize)
    , m_log2FFTSize(frame.m_log2FFTSize)
    , m_FFTSetup(frame.m_FFTSetup)
    , m_realData(frame.m_FFTSize)
    , m_imagData(frame.m_FFTSize)
{
    // Setup frame data
    m_frame.realp = m_realData.data();
    m_frame.imagp = m_imagData.data();

    // Copy/setup frame data
    memcpySpan(realData().span(), unsafeMakeSpan(frame.m_frame.realp, realData().size()));
    memcpySpan(imagData().span(), unsafeMakeSpan(frame.m_frame.imagp, imagData().size()));
}

FFTFrame::~FFTFrame() = default;

void FFTFrame::doFFT(std::span<const float> data)
{
    unsigned halfSize = m_FFTSize / 2;
    vDSP_ctoz(&reinterpretCastSpanStartTo<const DSPComplex>(data), 2, &m_frame, 1, halfSize);
    vDSP_fft_zrip(m_FFTSetup, &m_frame, 1, m_log2FFTSize, FFT_FORWARD);

    RELEASE_ASSERT(realData().size() >= halfSize);
    RELEASE_ASSERT(imagData().size() >= halfSize);

    // To provide the best possible execution speeds, the vDSP library's functions don't always adhere strictly
    // to textbook formulas for Fourier transforms, and must be scaled accordingly.
    // (See https://developer.apple.com/library/archive/documentation/Performance/Conceptual/vDSP_Programming_Guide/UsingFourierTransforms/UsingFourierTransforms.html#//apple_ref/doc/uid/TP40005147-CH3-SW5)
    // In the case of a Real forward Transform like above: RFimp = RFmath * 2 so we need to divide the output
    // by 2 to get the correct value.
    VectorMath::multiplyByScalar(realData().span().first(halfSize), 0.5, realData().span());
    VectorMath::multiplyByScalar(imagData().span().first(halfSize), 0.5, imagData().span());
}

void FFTFrame::doInverseFFT(std::span<float> data)
{
    vDSP_fft_zrip(m_FFTSetup, &m_frame, 1, m_log2FFTSize, FFT_INVERSE);
    vDSP_ztoc(&m_frame, 1, &reinterpretCastSpanStartTo<DSPComplex>(data), 2, m_FFTSize / 2);

    // Do final scaling so that x == IFFT(FFT(x))
    VectorMath::multiplyByScalar(data.first(m_FFTSize), 1.0f / m_FFTSize, data);
}

FFTSetup FFTFrame::fftSetupForSize(unsigned fftSize)
{
    auto pow2size = static_cast<size_t>(log2(fftSize));
    ASSERT(pow2size < kMaxFFTPow2Size);

    Locker locker { fftSetupsLock };
    auto& fftSetup = fftSetups().at(pow2size);
    if (!fftSetup)
        fftSetup = vDSP_create_fftsetup(pow2size, FFT_RADIX2);

    return fftSetup;
}

int FFTFrame::minFFTSize()
{
    return 1 << kMinFFTPow2Size;
}

int FFTFrame::maxFFTSize()
{
    return 1 << kMaxFFTPow2Size;
}

void FFTFrame::initialize()
{
}

} // namespace WebCore

#endif // #if OS(DARWIN)

#endif // ENABLE(WEB_AUDIO)
