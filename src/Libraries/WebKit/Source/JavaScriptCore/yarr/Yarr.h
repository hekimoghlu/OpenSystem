/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 17, 2024.
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

#include <limits.h>
#include "YarrErrorCode.h"

namespace JSC { namespace Yarr {

#define YarrStackSpaceForBackTrackInfoPatternCharacter 2 // Only for !fixed quantifiers.
#define YarrStackSpaceForBackTrackInfoCharacterClass 2 // Only for !fixed quantifiers.
#define YarrStackSpaceForBackTrackInfoBackReference 3
#define YarrStackSpaceForBackTrackInfoAlternative 1 // One per alternative.
#define YarrStackSpaceForBackTrackInfoParentheticalAssertion 1
#define YarrStackSpaceForBackTrackInfoParenthesesOnce 2
#define YarrStackSpaceForBackTrackInfoParenthesesTerminal 1
#define YarrStackSpaceForBackTrackInfoParentheses 4
#define YarrStackSpaceForDotStarEnclosure 1

static constexpr unsigned quantifyInfinite = UINT_MAX;
static constexpr uint64_t quantifyInfinite64 = std::numeric_limits<uint64_t>::max();
static constexpr unsigned offsetNoMatch = std::numeric_limits<unsigned>::max();

// The below limit restricts the number of "recursive" match calls in order to
// avoid spending exponential time on complex regular expressions.
static constexpr unsigned matchLimit = 100000000;

enum class MatchFrom { VMThread, CompilerThread };

enum class JSRegExpResult {
    Match = 1,
    NoMatch = 0,
    ErrorNoMatch = -1,
    JITCodeFailure = -2,
    ErrorHitLimit = -3,
    ErrorNoMemory = -4,
    ErrorInternal = -5,
};

enum class CharSize : uint8_t {
    Char8,
    Char16
};

enum class BuiltInCharacterClassID : unsigned {
    DigitClassID,
    SpaceClassID,
    WordClassID,
    DotClassID,
    BaseUnicodePropertyID,
};

enum class SpecificPattern : uint8_t {
    None,
    Atom,
    LeadingSpacesStar,
    LeadingSpacesPlus,
    TrailingSpacesStar,
    TrailingSpacesPlus,
};

struct BytecodePattern;

} } // namespace JSC::Yarr
