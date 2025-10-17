/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 10, 2023.
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

#if ENABLE(CONTENT_EXTENSIONS)

#include "ContentExtensionsDebugging.h"
#include <wtf/HashMap.h>
#include <wtf/HashSet.h>
#include <wtf/Vector.h>

namespace WebCore {

namespace ContentExtensions {

// A NFANode abstract the transition table out of a NFA state.

typedef Vector<uint64_t, 0, CrashOnOverflow, 1> ActionList;
typedef UncheckedKeyHashSet<unsigned, DefaultHash<unsigned>, WTF::UnsignedWithZeroKeyHashTraits<unsigned>> NFANodeIndexSet;
typedef UncheckedKeyHashMap<uint16_t, NFANodeIndexSet, DefaultHash<uint16_t>, WTF::UnsignedWithZeroKeyHashTraits<uint16_t>> NFANodeTransitions;

class NFANode {
public:
    NFANodeTransitions transitions;
    NFANodeIndexSet transitionsOnAnyCharacter;

    ActionList finalRuleIds;
};

} // namespace ContentExtensions
} // namespace WebCore

#endif // ENABLE(CONTENT_EXTENSIONS)
