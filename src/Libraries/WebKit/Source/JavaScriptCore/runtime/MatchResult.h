/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 7, 2022.
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

#include "UGPRPair.h"
#include <wtf/NotFound.h>
#include <wtf/PrintStream.h>

namespace JSC {

struct MatchResult {
    constexpr MatchResult() = default;

    ALWAYS_INLINE MatchResult(size_t start, size_t end)
        : start(start)
        , end(end)
    {
    }

    ALWAYS_INLINE MatchResult(UGPRPair match)
    {
        decodeResult(match, start, end);
    }

    ALWAYS_INLINE static constexpr MatchResult failed()
    {
        return MatchResult();
    }

    ALWAYS_INLINE explicit operator bool() const
    {
        return start != WTF::notFound;
    }

    ALWAYS_INLINE bool empty()
    {
        return start == end;
    }
    
    void dump(PrintStream&) const;

    size_t start { WTF::notFound };
    size_t end { 0 };
};

#if ENABLE(JIT)
static_assert(sizeof(UGPRPair) == 2 * sizeof(size_t), "https://bugs.webkit.org/show_bug.cgi?id=198518#c11");
static_assert(sizeof(MatchResult) == sizeof(UGPRPair), "Match result and UGPRPair should be the same size");
#endif

} // namespace JSC
