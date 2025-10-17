/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 10, 2025.
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

#include "SourceID.h"
#include <wtf/GenericHashKey.h>
#include <wtf/HashMap.h>
#include <wtf/HashTraits.h>
#include <wtf/Vector.h>

namespace JSC {

class FunctionHasExecutedCache {
public:
    struct FunctionRange {
        struct Hash {
            static unsigned hash(const FunctionRange& key) { return key.hash(); }
            static bool equal(const FunctionRange& a, const FunctionRange& b) { return a == b; }
            static constexpr bool safeToCompareToEmptyOrDeleted = false;
        };

        FunctionRange() {}
        friend bool operator==(const FunctionRange&, const FunctionRange&) = default;
        unsigned hash() const
        {
            return m_start * m_end;
        }

        unsigned m_start;
        unsigned m_end;
    };

    bool hasExecutedAtOffset(SourceID, unsigned offset);
    void insertUnexecutedRange(SourceID, unsigned start, unsigned end);
    void removeUnexecutedRange(SourceID, unsigned start, unsigned end);
    Vector<std::tuple<bool, unsigned, unsigned>> getFunctionRanges(SourceID);

private:
    using RangeMap = UncheckedKeyHashMap<GenericHashKey<FunctionRange, FunctionRange::Hash>, bool>;
    using SourceIDToRangeMap = UncheckedKeyHashMap<GenericHashKey<intptr_t>, RangeMap>;
    SourceIDToRangeMap m_rangeMap;
};

} // namespace JSC
