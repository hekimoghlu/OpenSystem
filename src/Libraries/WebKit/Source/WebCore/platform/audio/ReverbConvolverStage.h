/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 9, 2025.
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
#ifndef ReverbConvolverStage_h
#define ReverbConvolverStage_h

#include "AudioArray.h"
#include <memory>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class FFTFrame;
class ReverbAccumulationBuffer;
class ReverbConvolver;
class FFTConvolver;
class DirectConvolver;
    
// A ReverbConvolverStage represents the convolution associated with a sub-section of a large impulse response.
// It incorporates a delay line to account for the offset of the sub-section within the larger impulse response.
class ReverbConvolverStage final {
    WTF_MAKE_TZONE_ALLOCATED(ReverbConvolverStage);
public:
    // renderPhase is useful to know so that we can manipulate the pre versus post delay so that stages will perform
    // their heavy work (FFT processing) on different slices to balance the load in a real-time thread.
    ReverbConvolverStage(std::span<const float> impulseResponse, size_t reverbTotalLatency, size_t stageOffset, size_t stageLength, size_t fftSize, size_t renderPhase, size_t renderSliceSize, ReverbAccumulationBuffer*, float scale, bool directMode = false);
    ~ReverbConvolverStage();

    // WARNING: source.size() must be such that it evenly divides the delay buffer size (stage_offset).
    void process(std::span<const float> source);

    void processInBackground(ReverbConvolver* convolver, size_t framesToProcess);

    void reset();

    // Useful for background processing
    int inputReadIndex() const { return m_inputReadIndex; }

private:
    std::unique_ptr<FFTFrame> m_fftKernel;
    std::unique_ptr<FFTConvolver> m_fftConvolver;

    AudioFloatArray m_preDelayBuffer;

    ReverbAccumulationBuffer* m_accumulationBuffer;
    int m_accumulationReadIndex { 0 };
    int m_inputReadIndex { 0 };

    size_t m_preDelayLength;
    size_t m_postDelayLength;
    size_t m_preReadWriteIndex;
    size_t m_framesProcessed;

    AudioFloatArray m_temporaryBuffer;

    bool m_directMode;
    std::unique_ptr<AudioFloatArray> m_directKernel;
    std::unique_ptr<DirectConvolver> m_directConvolver;
};

} // namespace WebCore

#endif // ReverbConvolverStage_h
