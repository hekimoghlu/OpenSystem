/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 9, 2025.
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
#include "config.h"
#include "DFANode.h"

#include "DFA.h"
#include <wtf/HashMap.h>

#if ENABLE(CONTENT_EXTENSIONS)

namespace WebCore {

namespace ContentExtensions {

Vector<uint64_t> DFANode::actions(const DFA& dfa) const
{
    // FIXME: Use iterators instead of copying the Vector elements.
    return Vector<uint64_t>(m_actionsLength, [&](size_t i) {
        return dfa.actions[m_actionsStart + i];
    });
}

bool DFANode::containsTransition(char transition, const DFA& dfa) const
{
    // Called from DFAMinimizer, this loops though a maximum of 128 transitions, so it's not too slow.
    ASSERT(m_transitionsLength <= 128);
    for (unsigned i = m_transitionsStart; i < m_transitionsStart + m_transitionsLength; ++i) {
        if (dfa.transitionRanges[i].first <= transition
            && dfa.transitionRanges[i].last >= transition)
            return true;
    }
    return false;
}

void DFANode::kill(DFA& dfa)
{
    ASSERT(m_flags != IsKilled);
    m_flags = IsKilled; // Killed nodes don't have any other flags.

    // Invalidate the now-unused memory in the DFA to make finding bugs easier.
    for (unsigned i = m_transitionsStart; i < m_transitionsStart + m_transitionsLength; ++i) {
        dfa.transitionRanges[i] = { -1, -1 };
        dfa.transitionDestinations[i] = std::numeric_limits<uint32_t>::max();
    }
    for (unsigned i = m_actionsStart; i < m_actionsStart + m_actionsLength; ++i)
        dfa.actions[i] = std::numeric_limits<uint64_t>::max();

    m_actionsStart = 0;
    m_actionsLength = 0;
    m_transitionsStart = 0;
    m_transitionsLength = 0;
};

bool DFANode::canUseFallbackTransition(const DFA& dfa) const
{
    // Transitions can contain '\0' if the expression has a end-of-line marker.
    // Fallback transitions cover 1-127. We have to be careful with the first.

    IterableConstRange iterableTransitions = transitions(dfa);
    auto iterator = iterableTransitions.begin();
    auto end = iterableTransitions.end();
    if (iterator == end)
        return false;

    char lastSeenCharacter = 0;
    if (!iterator.first()) {
        lastSeenCharacter = iterator.last();
        if (lastSeenCharacter == 127)
            return true;
        ++iterator;
    }

    for (;iterator != end; ++iterator) {
        ASSERT(iterator.first() > lastSeenCharacter);
        if (iterator.first() != lastSeenCharacter + 1)
            return false;

        if (iterator.range().last == 127)
            return true;
        lastSeenCharacter = iterator.last();
    }
    return false;
}

uint32_t DFANode::bestFallbackTarget(const DFA& dfa) const
{
    ASSERT(canUseFallbackTransition(dfa));

    UncheckedKeyHashMap<uint32_t, unsigned, DefaultHash<uint32_t>, WTF::UnsignedWithZeroKeyHashTraits<uint32_t>> histogram;

    IterableConstRange iterableTransitions = transitions(dfa);
    auto iterator = iterableTransitions.begin();
    auto end = iterableTransitions.end();
    ASSERT_WITH_MESSAGE(iterator != end, "An empty range list cannot use a fallback transition.");

    if (!iterator.first() && !iterator.last())
        ++iterator;
    ASSERT_WITH_MESSAGE(iterator != end, "An empty range list matching only zero cannot use a fallback transition.");

    uint32_t bestTarget = iterator.target();
    unsigned bestTargetScore = !iterator.range().first ? iterator.range().size() - 1 : iterator.range().size();
    histogram.add(bestTarget, bestTargetScore);
    ++iterator;

    for (;iterator != end; ++iterator) {
        unsigned rangeSize = iterator.range().size();
        uint32_t target = iterator.target();
        auto addResult = histogram.add(target, rangeSize);
        if (!addResult.isNewEntry)
            addResult.iterator->value += rangeSize;
        if (addResult.iterator->value > bestTargetScore) {
            bestTargetScore = addResult.iterator->value;
            bestTarget = target;
        }
    }
    return bestTarget;
}

} // namespace ContentExtensions

} // namespace WebCore

#endif // ENABLE(CONTENT_EXTENSIONS)
