/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 18, 2024.
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
#ifndef SincResampler_h
#define SincResampler_h

#include "AudioArray.h"
#include "AudioSourceProvider.h"
#include <span>
#include <wtf/Function.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

// SincResampler is a high-quality sample-rate converter.

class SincResampler final {
    WTF_MAKE_TZONE_ALLOCATED(SincResampler);
public:
    // scaleFactor == sourceSampleRate / destinationSampleRate
    // requestFrames controls the size in frames of the buffer requested by each provideInput() call.
    SincResampler(double scaleFactor, unsigned requestFrames, Function<void(std::span<float> buffer, size_t framesToProcess)>&& provideInput);
    
    size_t chunkSize() const { return m_chunkSize; }

    // Processes samples in `source` to produce source.size() / scaleFactor frames in `destination`.
    WEBCORE_EXPORT static void processBuffer(std::span<const float> source, std::span<float> destination, double scaleFactor);

    // Process with provideInput callback function for streaming applications.
    void process(std::span<float> destination, size_t framesToProcess);

private:
    void initializeKernel();
    void updateRegions(bool isSecondLoad);

    float convolve(std::span<const float> inputP, std::span<const float> k1, std::span<const float> k2, float kernelInterpolationFactor);
    
    double m_scaleFactor;

    // m_kernelStorage has m_numberOfKernelOffsets kernels back-to-back, each of size m_kernelSize.
    // The kernel offsets are sub-sample shifts of a windowed sinc() shifted from 0.0 to 1.0 sample.
    AudioFloatArray m_kernelStorage;
    
    // m_virtualSourceIndex is an index on the source input buffer with sub-sample precision.
    // It must be double precision to avoid drift.
    double m_virtualSourceIndex { 0 };
    
    // This is the number of destination frames we generate per processing pass on the buffer.
    unsigned m_requestFrames;

    Function<void(std::span<float> buffer, size_t framesToProcess)> m_provideInput;

    // The number of source frames processed per pass.
    size_t m_blockSize { 0 };

    size_t m_chunkSize { 0 };

    // Source is copied into this buffer for each processing pass.
    AudioFloatArray m_inputBuffer;

    // Spans to the various regions inside |m_inputBuffer|. See the diagram at
    // the top of the .cpp file for more information.
    std::span<float> m_r0;
    const std::span<float> m_r1;
    const std::span<float> m_r2;
    std::span<float> m_r3;
    std::span<float> m_r4;

    // The buffer is primed once at the very beginning of processing.
    bool m_isBufferPrimed { false };
};

} // namespace WebCore

#endif // SincResampler_h
