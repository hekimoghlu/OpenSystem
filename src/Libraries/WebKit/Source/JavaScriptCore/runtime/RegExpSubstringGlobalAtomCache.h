/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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

#include "MatchResult.h"
#include "SlotVisitorMacros.h"
#include "WriteBarrier.h"

namespace JSC {

class JSString;
class JSRopeString;
class RegExp;

// This class is used to track the cached results of the last match
// on a substring with global and atomic pattern regexp. It's
// looking for the code pattern:
//
//     str.substring(0, 50).match(regExp);
//     str.substring(0, 60).match(regExp);
class RegExpSubstringGlobalAtomCache {
public:
    JSValue collectMatches(JSGlobalObject*, JSRopeString* substring, RegExp*);

    DECLARE_VISIT_AGGREGATE;

private:
    WriteBarrier<JSString> m_lastSubstringBase;
    unsigned m_lastSubstringOffset;
    unsigned m_lastSubstringLength;

    WriteBarrier<RegExp> m_lastRegExp;
    size_t m_lastNumberOfMatches { 0 };
    size_t m_lastMatchEnd { 0 };
    MatchResult m_lastResult { };
};

} // namespace JSC
