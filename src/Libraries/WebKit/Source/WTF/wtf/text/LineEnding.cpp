/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 12, 2022.
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
#include <wtf/text/LineEnding.h>

#include <wtf/Vector.h>

namespace WTF {

Vector<uint8_t> normalizeLineEndingsToLF(Vector<uint8_t>&& vector)
{
    size_t inputIndex = 0;
    size_t outputIndex = 0;
    while (inputIndex < vector.size()) {
        auto character = vector[inputIndex++];
        if (character == '\r') {
            // Turn CRLF and CR into LF.
            if (inputIndex < vector.size() && vector[inputIndex] == '\n')
                ++inputIndex;
            vector[outputIndex++] = '\n';
        } else {
            // Leave other characters alone.
            vector[outputIndex++] = character;
        }
    }
    vector.shrink(outputIndex);
    return vector;
}

Vector<uint8_t> normalizeLineEndingsToCRLF(Vector<uint8_t>&& source)
{
    size_t sourceIndex = 0;
    size_t resultLength = 0;
    while (sourceIndex < source.size()) {
        auto character = source[sourceIndex++];
        if (character == '\r') {
            // Turn CR or CRLF into CRLF;
            if (sourceIndex < source.size() && source[sourceIndex] == '\n')
                ++sourceIndex;
            resultLength += 2;
        } else if (character == '\n') {
            // Turn LF into CRLF.
            resultLength += 2;
        } else {
            // Leave other characters alone.
            resultLength += 1;
        }
    }

    if (resultLength == source.size())
        return source;

    Vector<uint8_t> result(resultLength);
    sourceIndex = 0;
    size_t resultIndex = 0;
    while (sourceIndex < source.size()) {
        auto character = source[sourceIndex++];
        if (character == '\r') {
            // Turn CR or CRLF into CRLF;
            if (sourceIndex < source.size() && source[sourceIndex] == '\n')
                ++sourceIndex;
            result[resultIndex++] = '\r';
            result[resultIndex++] = '\n';
        } else if (character == '\n') {
            // Turn LF into CRLF.
            result[resultIndex++] = '\r';
            result[resultIndex++] = '\n';
        } else {
            // Leave other characters alone.
            result[resultIndex++] = character;
        }
    }
    ASSERT(resultIndex == resultLength);
    return result;
}

Vector<uint8_t> normalizeLineEndingsToNative(Vector<uint8_t>&& from)
{
#if OS(WINDOWS)
    return normalizeLineEndingsToCRLF(WTFMove(from));
#else
    return normalizeLineEndingsToLF(WTFMove(from));
#endif
}

} // namespace WTF
