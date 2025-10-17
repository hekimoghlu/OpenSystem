/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 31, 2023.
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
#ifndef FFTConvolver_h
#define FFTConvolver_h

#include "AudioArray.h"
#include "FFTFrame.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class FFTConvolver final {
    WTF_MAKE_TZONE_ALLOCATED(FFTConvolver);
    WTF_MAKE_NONCOPYABLE(FFTConvolver);
public:
    // fftSize must be a power of two
    FFTConvolver(size_t fftSize);

    // For now, with multiple calls to Process(), source.size() MUST add up EXACTLY to fftSize / 2
    //
    // FIXME: Later, we can do more sophisticated buffering to relax this requirement...
    //
    // The input to output latency is equal to fftSize / 2
    //
    // Processing in-place is allowed...
    void process(FFTFrame* fftKernel, std::span<const float> source, std::span<float> destination);

    void reset();

    size_t fftSize() const { return m_frame.fftSize(); }

private:
    FFTFrame m_frame;

    // Buffer input until we get fftSize / 2 samples then do an FFT
    size_t m_readWriteIndex { 0 };
    AudioFloatArray m_inputBuffer;

    // Stores output which we read a little at a time
    AudioFloatArray m_outputBuffer;

    // Saves the 2nd half of the FFT buffer, so we can do an overlap-add with the 1st half of the next one
    AudioFloatArray m_lastOverlapBuffer;
};

} // namespace WebCore

#endif // FFTConvolver_h
