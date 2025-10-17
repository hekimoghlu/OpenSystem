/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 25, 2022.
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
#ifndef ReverbAccumulationBuffer_h
#define ReverbAccumulationBuffer_h

#include "AudioArray.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

// ReverbAccumulationBuffer is a circular delay buffer with one client reading from it and multiple clients
// writing/accumulating to it at different delay offsets from the read position.  The read operation will zero the memory
// just read from the buffer, so it will be ready for accumulation the next time around.
class ReverbAccumulationBuffer final {
    WTF_MAKE_TZONE_ALLOCATED(ReverbAccumulationBuffer);
public:
    explicit ReverbAccumulationBuffer(size_t length);

    // This will read from, then clear-out numberOfFrames
    void readAndClear(std::span<float> destination, size_t numberOfFrames);

    // Each ReverbConvolverStage will accumulate its output at the appropriate delay from the read position.
    // We need to pass in and update readIndex here, since each ReverbConvolverStage may be running in
    // a different thread than the realtime thread calling ReadAndClear() and maintaining m_readIndex
    // Returns the writeIndex where the accumulation took place
    int accumulate(std::span<float> source, size_t numberOfFrames, int* readIndex, size_t delayFrames);

    size_t readIndex() const { return m_readIndex; }
    void updateReadIndex(int* readIndex, size_t numberOfFrames) const;

    size_t readTimeFrame() const { return m_readTimeFrame; }

    void reset();

private:
    AudioFloatArray m_buffer;
    size_t m_readIndex { 0 };
    size_t m_readTimeFrame { 0 }; // for debugging (frame on continuous timeline)
};

} // namespace WebCore

#endif // ReverbAccumulationBuffer_h
