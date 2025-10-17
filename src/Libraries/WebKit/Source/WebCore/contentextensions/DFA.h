/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 19, 2021.
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
#include "DFANode.h"
#include <wtf/Vector.h>

namespace WebCore {

namespace ContentExtensions {

// The DFA abstract a partial DFA graph in a compact form.
struct WEBCORE_EXPORT DFA {
    static DFA empty();

    void minimize();
    unsigned graphSize() const;
    size_t memoryUsed() const;

#if CONTENT_EXTENSIONS_STATE_MACHINE_DEBUGGING
    void debugPrintDot() const;
#endif

    Vector<DFANode, 0, ContentExtensionsOverflowHandler> nodes;
    Vector<uint64_t, 0, ContentExtensionsOverflowHandler> actions;
    Vector<CharRange, 0, ContentExtensionsOverflowHandler> transitionRanges;
    Vector<uint32_t, 0, ContentExtensionsOverflowHandler> transitionDestinations;
    unsigned root { 0 };
};

inline const CharRange& DFANode::ConstRangeIterator::range() const
{
    return dfa.transitionRanges[position];
}

inline uint32_t DFANode::ConstRangeIterator::target() const
{
    return dfa.transitionDestinations[position];
}

inline const CharRange& DFANode::RangeIterator::range() const
{
    return dfa.transitionRanges[position];
}

inline uint32_t DFANode::RangeIterator::target() const
{
    return dfa.transitionDestinations[position];
}

inline void DFANode::RangeIterator::resetTarget(uint32_t newTarget)
{
    dfa.transitionDestinations[position] = newTarget;
}

} // namespace ContentExtensions
} // namespace WebCore

#endif // ENABLE(CONTENT_EXTENSIONS)
