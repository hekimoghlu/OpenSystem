/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 10, 2025.
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
#include "DFA.h"

#if ENABLE(CONTENT_EXTENSIONS)

#include "DFAMinimizer.h"
#include <wtf/DataLog.h>

namespace WebCore {

namespace ContentExtensions {

DFA DFA::empty()
{
    DFA newDFA;
    newDFA.nodes.append(DFANode());
    return newDFA;
}

size_t DFA::memoryUsed() const
{
    return sizeof(DFA)
        + actions.capacity() * sizeof(uint64_t)
        + transitionRanges.capacity() * sizeof(CharRange)
        + transitionDestinations.capacity() * sizeof(uint32_t)
        + nodes.capacity() * sizeof(DFANode);
}

void DFA::minimize()
{
    DFAMinimizer::minimize(*this);
}

unsigned DFA::graphSize() const
{
    unsigned count = 0;
    for (const DFANode& node : nodes) {
        if (!node.isKilled())
            ++count;
    }
    return count;
}

#if CONTENT_EXTENSIONS_STATE_MACHINE_DEBUGGING
static void printRange(bool firstRange, char rangeStart, char rangeEnd)
{
    if (!firstRange)
        dataLogF(", ");
    if (rangeStart == rangeEnd) {
        if (rangeStart == '"' || rangeStart == '\\')
            dataLogF("\\%c", rangeStart);
        else if (rangeStart >= '!' && rangeStart <= '~')
            dataLogF("%c", rangeStart);
        else
            dataLogF("\\\\%d", rangeStart);
    } else
        dataLogF("\\\\%d-\\\\%d", rangeStart, rangeEnd);
}

static void printTransitions(const DFA& dfa, unsigned sourceNodeId)
{
    const DFANode& sourceNode = dfa.nodes[sourceNodeId];
    auto transitions = sourceNode.transitions(dfa);

    if (transitions.begin() == transitions.end())
        return;

    UncheckedKeyHashMap<unsigned, Vector<uint16_t>, DefaultHash<unsigned>, WTF::UnsignedWithZeroKeyHashTraits<unsigned>> transitionsPerTarget;

    // First, we build the list of transitions coming to each target node.
    for (const auto& transition : transitions) {
        unsigned target = transition.target();
        transitionsPerTarget.add(target, Vector<uint16_t>());

        for (unsigned offset = 0; offset < transition.range().size(); ++offset)
            transitionsPerTarget.find(target)->value.append(transition.first() + offset);
    }

    // Then we go over each one an display the ranges one by one.
    for (const auto& transitionPerTarget : transitionsPerTarget) {
        dataLogF("        %d -> %d [label=\"", sourceNodeId, transitionPerTarget.key);

        Vector<uint16_t> incomingCharacters = transitionPerTarget.value;
        std::sort(incomingCharacters.begin(), incomingCharacters.end());

        char rangeStart = incomingCharacters.first();
        char rangeEnd = rangeStart;
        bool first = true;
        for (unsigned sortedTransitionIndex = 1; sortedTransitionIndex < incomingCharacters.size(); ++sortedTransitionIndex) {
            char nextChar = incomingCharacters[sortedTransitionIndex];
            if (nextChar == rangeEnd+1) {
                rangeEnd = nextChar;
                continue;
            }
            printRange(first, rangeStart, rangeEnd);
            rangeStart = rangeEnd = nextChar;
            first = false;
        }
        printRange(first, rangeStart, rangeEnd);

        dataLogF("\"];\n");
    }
}

void DFA::debugPrintDot() const
{
    dataLogF("digraph DFA_Transitions {\n");
    dataLogF("    rankdir=LR;\n");
    dataLogF("    node [shape=circle];\n");
    dataLogF("    {\n");
    for (unsigned i = 0; i < nodes.size(); ++i) {
        if (nodes[i].isKilled())
            continue;

        dataLogF("         %d [label=<Node %d", i, i);
        const Vector<uint64_t>& actions = nodes[i].actions(*this);
        if (!actions.isEmpty()) {
            dataLogF("<BR/>Actions: ");
            for (unsigned actionIndex = 0; actionIndex < actions.size(); ++actionIndex) {
                if (actionIndex)
                    dataLogF(", ");
                dataLogF("%" PRIu64, actions[actionIndex]);
            }
        }
        dataLogF(">]");

        if (!actions.isEmpty())
            dataLogF(" [shape=doublecircle]");

        dataLogF(";\n");
    }
    dataLogF("    }\n");

    dataLogF("    {\n");
    for (unsigned i = 0; i < nodes.size(); ++i)
        printTransitions(*this, i);

    dataLogF("    }\n");
    dataLogF("}\n");
}
#endif

} // namespace ContentExtensions

} // namespace WebCore

#endif // ENABLE(CONTENT_EXTENSIONS)
