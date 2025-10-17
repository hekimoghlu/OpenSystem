/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 11, 2024.
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

namespace WGSL {

struct SourcePosition {
    unsigned line;
    unsigned lineOffset;
    unsigned offset;
};

struct SourceSpan {
    // FIXME: we could possibly skip lineOffset and recompute it only when trying to show an error
    // This would shrink the AST size by 32 bits per AST node, at the cost of a bit of code complexity in the error toString function.
    unsigned line;
    unsigned lineOffset;
    unsigned offset;
    unsigned length;

    static constexpr SourceSpan empty() { return { 0, 0, 0, 0 }; }

    constexpr SourceSpan(unsigned line, unsigned lineOffset, unsigned offset, unsigned length)
        : line(line)
        , lineOffset(lineOffset)
        , offset(offset)
        , length(length)
    { }

    constexpr SourceSpan(SourcePosition start, SourcePosition end)
        : SourceSpan(start.line, start.lineOffset, start.offset, end.offset - start.offset)
    { }

    friend constexpr bool operator==(const SourceSpan&, const SourceSpan&) = default;
};

}
