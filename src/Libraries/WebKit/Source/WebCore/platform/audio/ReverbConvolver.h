/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 15, 2022.
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
#ifndef ReverbConvolver_h
#define ReverbConvolver_h

#include "ReverbAccumulationBuffer.h"
#include "ReverbConvolverStage.h"
#include "ReverbInputBuffer.h"
#include <memory>
#include <wtf/Condition.h>
#include <wtf/Lock.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Threading.h>
#include <wtf/Vector.h>

namespace WebCore {

class AudioChannel;

class ReverbConvolver final {
    WTF_MAKE_TZONE_ALLOCATED(ReverbConvolver);
    WTF_MAKE_NONCOPYABLE(ReverbConvolver);
public:
    // maxFFTSize can be adjusted (from say 2048 to 32768) depending on how much precision is necessary.
    // For certain tweaky de-convolving applications the phase errors add up quickly and lead to non-sensical results with
    // larger FFT sizes and single-precision floats.  In these cases 2048 is a good size.
    // If not doing multi-threaded convolution, then should not go > 8192.
    ReverbConvolver(AudioChannel* impulseResponse, size_t renderSliceSize, size_t maxFFTSize, size_t convolverRenderPhase, bool useBackgroundThreads, float scale);
    ~ReverbConvolver();

    void process(const AudioChannel* sourceChannel, AudioChannel* destinationChannel, size_t framesToProcess);
    void reset();

    size_t impulseResponseLength() const { return m_impulseResponseLength; }

    ReverbInputBuffer* inputBuffer() { return &m_inputBuffer; }

    bool useBackgroundThreads() const { return m_useBackgroundThreads; }

    size_t latencyFrames() const;
private:
    void backgroundThreadEntry();

    Vector<std::unique_ptr<ReverbConvolverStage>> m_stages;
    Vector<std::unique_ptr<ReverbConvolverStage>> m_backgroundStages;
    size_t m_impulseResponseLength;

    ReverbAccumulationBuffer m_accumulationBuffer;

    // One or more background threads read from this input buffer which is fed from the realtime thread.
    ReverbInputBuffer m_inputBuffer;

    // First stage will be of size m_minFFTSize.  Each next stage will be twice as big until we hit m_maxFFTSize.
    size_t m_minFFTSize;
    size_t m_maxFFTSize;

    // But don't exceed this size in the real-time thread (if we're doing background processing).
    size_t m_maxRealtimeFFTSize;

    // Background thread and synchronization
    bool m_useBackgroundThreads;
    RefPtr<Thread> m_backgroundThread;
    bool m_wantsToExit { false };
    bool m_moreInputBuffered { false };
    mutable Lock m_backgroundThreadLock;
    mutable Condition m_backgroundThreadConditionVariable;
};

} // namespace WebCore

#endif // ReverbConvolver_h
