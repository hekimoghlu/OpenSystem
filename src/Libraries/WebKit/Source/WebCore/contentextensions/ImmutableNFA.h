/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 27, 2025.
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
#include <wtf/Vector.h>

namespace WebCore {

namespace ContentExtensions {

template <typename CharacterType>
struct ImmutableRange {
    uint32_t targetStart;
    uint32_t targetEnd;
    CharacterType first;
    CharacterType last;
};

struct ImmutableNFANode {
    uint32_t rangesStart { 0 };
    uint32_t rangesEnd { 0 };
    uint32_t epsilonTransitionTargetsStart { 0 };
    uint32_t epsilonTransitionTargetsEnd { 0 };
    uint32_t actionStart { 0 };
    uint32_t actionEnd { 0 };
};

template <typename CharacterType, typename ActionType>
struct ImmutableNFA {
    Vector<ImmutableNFANode, 0, ContentExtensionsOverflowHandler> nodes;
    Vector<ImmutableRange<CharacterType>, 0, ContentExtensionsOverflowHandler> transitions;
    Vector<uint32_t, 0, ContentExtensionsOverflowHandler> targets;
    Vector<uint32_t, 0, ContentExtensionsOverflowHandler> epsilonTransitionsTargets;
    Vector<ActionType, 0, ContentExtensionsOverflowHandler> actions;

    void clear()
    {
        nodes.clear();
        transitions.clear();
        targets.clear();
        epsilonTransitionsTargets.clear();
        actions.clear();
    }

    size_t memoryUsed() const
    {
        return nodes.capacity() * sizeof(ImmutableNFANode)
            + transitions.capacity() * sizeof(ImmutableRange<CharacterType>)
            + targets.capacity() * sizeof(uint32_t)
            + epsilonTransitionsTargets.capacity() * sizeof(uint32_t)
            + actions.capacity() * sizeof(ActionType);
    }
};

} // namespace ContentExtensions
} // namespace WebCore

#endif // ENABLE(CONTENT_EXTENSIONS)
