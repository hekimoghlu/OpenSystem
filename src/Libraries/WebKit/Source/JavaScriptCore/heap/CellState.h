/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 6, 2022.
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

#include <wtf/Assertions.h>

namespace JSC {

// The CellState of a cell is a kind of hint about what the state of the cell is.
enum class CellState : uint8_t {
    // The object is either currently being scanned, or it has finished being scanned, or this
    // is a full collection and it's actually a white object (you'd know because its mark bit
    // would be clear).
    PossiblyBlack = 0,
    
    // The object is in eden. During GC, this means that the object has not been marked yet.
    DefinitelyWhite = 1,

    // This sorta means that the object is grey - i.e. it will be scanned. Or it could be white
    // during a full collection if its mark bit is clear. That would happen if it had been black,
    // got barriered, and we did a full collection.
    PossiblyGrey = 2
};

static constexpr unsigned blackThreshold = 0; // x <= blackThreshold means x is PossiblyOldOrBlack.
static constexpr unsigned tautologicalThreshold = 100; // x <= tautologicalThreshold is always true.

inline bool isWithinThreshold(CellState cellState, unsigned threshold)
{
    return static_cast<unsigned>(cellState) <= threshold;
}

} // namespace JSC
