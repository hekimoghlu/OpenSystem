/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 4, 2022.
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

#if ENABLE(AX_THREAD_TEXT_APIS)

#include <wtf/text/MakeString.h>
#include <wtf/text/TextStream.h>

namespace WebCore {

class RenderBlock;

struct AXTextRunLineID {
    // Do not dereference, for comparison to other AXTextRunLineIDs only.
    void* containingBlock { nullptr };
    size_t lineIndex { 0 };

    AXTextRunLineID() = default;
    AXTextRunLineID(void* containingBlock, size_t lineIndex)
        : containingBlock(containingBlock)
        , lineIndex(lineIndex)
    { }
    bool operator==(const AXTextRunLineID&) const = default;
    operator bool() const { return containingBlock; }
    String debugDescription() const
    {
        TextStream stream;
        stream << "LineID " << containingBlock << " " << lineIndex;
        return stream.release();
    }
};

struct AXTextRun {
    // The line index of this run within the context of the containing RenderBlock of the main-thread AX object.
    size_t lineIndex;
    String text;

    AXTextRun(size_t lineIndex, String&& text)
        : lineIndex(lineIndex)
        , text(WTFMove(text))
    { }

    String debugDescription(void* containingBlock) const
    {
        AXTextRunLineID lineID = { containingBlock, lineIndex };
        return makeString(lineID.debugDescription(), ": |"_s, makeStringByReplacingAll(text, '\n', "{newline}"_s), "|(len "_s, text.length(), ")"_s);
    }

    // Convenience methods for TextUnit movement.
    bool startsWithLineBreak() const { return text.startsWith('\n'); }
    bool endsWithLineBreak() const { return text.endsWith('\n'); }
};

struct AXTextRuns {
    // The containing block for the text runs. This is required because based on the structure
    // of the AX tree, text runs for different objects can have the same line index but different
    // containing blocks, meaning they are rendered on different lines.
    // Do not de-reference. Use for comparison purposes only.
    void* containingBlock { nullptr };
    Vector<AXTextRun> runs;
    bool containsOnlyASCII { true };

    AXTextRuns() = default;
    AXTextRuns(RenderBlock* containingBlock, Vector<AXTextRun>&& textRuns)
        : containingBlock(containingBlock)
        , runs(WTFMove(textRuns))
    {
        for (const auto& run : runs) {
            if (!run.text.containsOnlyASCII()) {
                containsOnlyASCII = false;
                break;
            }
        }
    }
    String debugDescription() const;

    size_t size() const { return runs.size(); }
    const AXTextRun& at(size_t index) const
    {
        return (*this)[index];
    }
    const AXTextRun& operator[](size_t index) const
    {
        RELEASE_ASSERT(index < runs.size());
        return runs[index];
    }

    unsigned runLength(size_t index) const
    {
        RELEASE_ASSERT(index < runs.size());
        // Runs should have a non-zero length. This is important because several parts of AXTextMarker rely on this assumption.
        RELEASE_ASSERT(runs[index].text.length());
        return runs[index].text.length();
    }
    unsigned lastRunLength() const
    {
        if (runs.isEmpty())
            return 0;
        return runs[runs.size() - 1].text.length();
    }
    unsigned totalLength() const
    {
        unsigned size = runs.size();
        return size ? runLengthSumTo(size - 1) : 0;
    }
    unsigned runLengthSumTo(size_t index) const;

    size_t indexForOffset(unsigned textOffset) const;
    AXTextRunLineID lineIDForOffset(unsigned textOffset) const;
    AXTextRunLineID lineID(size_t index) const
    {
        RELEASE_ASSERT(index < runs.size());
        return { containingBlock, runs[index].lineIndex };
    }
    String substring(unsigned start, unsigned length = StringImpl::MaxLength) const;
    String toString() const { return substring(0); }
};

} // namespace WebCore
#endif // ENABLE(AX_THREAD_TEXT_APIS)
