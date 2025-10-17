/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 15, 2022.
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
#ifndef ReverbInputBuffer_h
#define ReverbInputBuffer_h

#include "AudioArray.h"

#include <wtf/TZoneMalloc.h>

namespace WebCore {

// ReverbInputBuffer is used to buffer input samples for deferred processing by the background threads.
class ReverbInputBuffer final {
    WTF_MAKE_TZONE_ALLOCATED(ReverbInputBuffer);
public:
    explicit ReverbInputBuffer(size_t length);

    // The realtime audio thread keeps writing samples here.
    // The assumption is that the buffer's length is evenly divisible by source.size() (for nearly all cases this will be fine).
    // FIXME: remove source.size() restriction...
    void write(std::span<const float> source);

    // Background threads can call this to check if there's anything to read...
    size_t writeIndex() const { return m_writeIndex; }

    // The individual background threads read here (and hope that they can keep up with the buffer writing).
    // readIndex is updated with the next readIndex to read from...
    // The assumption is that the buffer's length is evenly divisible by numberOfFrames.
    // FIXME: remove numberOfFrames restriction...
    std::span<float> directReadFrom(int* readIndex, size_t numberOfFrames);

    void reset();

private:
    AudioFloatArray m_buffer;
    size_t m_writeIndex { 0 };
};

} // namespace WebCore

#endif // ReverbInputBuffer_h
