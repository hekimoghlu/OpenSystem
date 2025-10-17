/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 14, 2023.
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
#pragma once

#if ENABLE(MHTML)

#include "SharedBuffer.h"
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class WEBCORE_EXPORT SharedBufferChunkReader {
public:
    SharedBufferChunkReader(FragmentedSharedBuffer*, const Vector<char>& separator);
    SharedBufferChunkReader(FragmentedSharedBuffer*, const char* separator);

    void setSeparator(const Vector<char>&);
    void setSeparator(const char*);

    // Returns false when the end of the buffer was reached.
    bool nextChunk(Vector<uint8_t>& data, bool includeSeparator = false);

    // Returns a null string when the end of the buffer has been reached.
    String nextChunkAsUTF8StringWithLatin1Fallback(bool includeSeparator = false);

    // Reads size bytes at the current location in the buffer, without changing the buffer position.
    // Returns the number of bytes read. That number might be less than the specified size if the end of the buffer was reached.
    size_t peek(Vector<uint8_t>&, size_t);

private:
    FragmentedSharedBuffer::DataSegmentVector::const_iterator m_iteratorCurrent;
    const FragmentedSharedBuffer::DataSegmentVector::const_iterator m_iteratorEnd;
    const uint8_t* m_segment { nullptr };
    size_t m_segmentIndex { 0 };
    Vector<char> m_separator { false };
    size_t m_separatorIndex { 0 };
};

}

#endif
