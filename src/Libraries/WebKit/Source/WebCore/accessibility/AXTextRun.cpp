/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 23, 2022.
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
#include "AXTextRun.h"

#if ENABLE(AX_THREAD_TEXT_APIS)

#include <wtf/text/MakeString.h>

namespace WebCore {

String AXTextRuns::debugDescription() const
{
    return makeString('[', interleave(runs, [&](auto& run) { return run.debugDescription(containingBlock); }, ", "_s), ']');
}

size_t AXTextRuns::indexForOffset(unsigned textOffset) const
{
    size_t cumulativeLength = 0;
    for (size_t i = 0; i < runs.size(); i++) {
        cumulativeLength += runLength(i);
        if (cumulativeLength >= textOffset)
            return i;
    }
    return notFound;
}

AXTextRunLineID AXTextRuns::lineIDForOffset(unsigned textOffset) const
{
    size_t runIndex = indexForOffset(textOffset);
    return runIndex == notFound ? AXTextRunLineID() : lineID(runIndex);
}

unsigned AXTextRuns::runLengthSumTo(size_t index) const
{
    unsigned length = 0;
    for (size_t i = 0; i <= index && i < runs.size(); i++)
        length += runLength(i);
    return length;
}

String AXTextRuns::substring(unsigned start, unsigned length) const
{
    if (!length)
        return emptyString();

    StringBuilder result;
    size_t charactersSeen = 0;
    auto remaining = [&] () {
        return result.length() >= length ? 0 : length - result.length();
    };
    for (unsigned i = 0; i < runs.size() && result.length() < length; i++) {
        size_t runLength = this->runLength(i);
        if (charactersSeen >= start) {
            // The start points entirely within bounds of this run.
            result.append(runs[i].text.left(remaining()));
        } else if (charactersSeen + runLength > start) {
            // start points somewhere in the middle of the current run, collect part of the text.
            unsigned startInRun = start - charactersSeen;
            RELEASE_ASSERT(startInRun < runLength);
            result.append(runs[i].text.substring(startInRun, remaining()));
        }
        // If charactersSeen + runLength == start, the start points to the end of the run, and there is no text to gather.

        charactersSeen += runLength;
    }
    return result.toString();
}

} // namespace WebCore
#endif // ENABLE(AX_THREAD_TEXT_APIS)
