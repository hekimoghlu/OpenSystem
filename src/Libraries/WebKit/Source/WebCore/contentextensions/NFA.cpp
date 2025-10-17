/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 31, 2022.
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
#include "NFA.h"

#if ENABLE(CONTENT_EXTENSIONS)

#include <wtf/ASCIICType.h>
#include <wtf/DataLog.h>

namespace WebCore {

namespace ContentExtensions {

#if CONTENT_EXTENSIONS_STATE_MACHINE_DEBUGGING
void NFA::debugPrintDot() const
{
    dataLogF("digraph NFA_Transitions {\n");
    dataLogF("    rankdir=LR;\n");
    dataLogF("    node [shape=circle];\n");
    dataLogF("    {\n");

    for (unsigned i = 0; i < nodes.size(); ++i) {
        dataLogF("         %d [label=<Node %d", i, i);

        const auto& node = nodes[i];

        if (node.actionStart != node.actionEnd) {
            dataLogF("<BR/>(Final: ");
            bool isFirst = true;
            for (unsigned actionIndex = node.actionStart; actionIndex < node.actionEnd; ++actionIndex) {
                if (!isFirst)
                    dataLogF(", ");
                dataLogF("%" PRIu64, actions[actionIndex]);
                isFirst = false;
            }
            dataLogF(")");
        }
        dataLogF(">]");

        if (node.actionStart != node.actionEnd)
            dataLogF(" [shape=doublecircle]");
        dataLogF(";\n");
    }
    dataLogF("    }\n");

    dataLogF("    {\n");
    for (unsigned i = 0; i < nodes.size(); ++i) {
        const auto& node = nodes[i];

        UncheckedKeyHashMap<uint32_t, Vector<uint32_t>, DefaultHash<unsigned>, WTF::UnsignedWithZeroKeyHashTraits<unsigned>> transitionsPerTarget;

        for (uint32_t transitionIndex = node.rangesStart; transitionIndex < node.rangesEnd; ++transitionIndex) {
            const ImmutableCharRange& range = transitions[transitionIndex];
            for (uint32_t targetIndex = range.targetStart; targetIndex < range.targetEnd; ++targetIndex) {
                uint32_t target = targets[targetIndex];
                auto addResult = transitionsPerTarget.add(target, Vector<uint32_t>());
                addResult.iterator->value.append(transitionIndex);
            }
        }

        for (const auto& slot : transitionsPerTarget) {
            dataLogF("        %d -> %d [label=\"", i, slot.key);

            bool isFirst = true;
            for (uint32_t rangeIndex : slot.value) {
                if (!isFirst)
                    dataLogF(", ");
                else
                    isFirst = false;

                const ImmutableCharRange& range = transitions[rangeIndex];
                if (range.first == range.last) {
                    if (isASCIIPrintable(range.first))
                    dataLogF("%c", range.first);
                    else
                    dataLogF("%d", range.first);
                } else {
                    if (isASCIIPrintable(range.first) && isASCIIPrintable(range.last))
                    dataLogF("%c-%c", range.first, range.last);
                    else
                    dataLogF("%d-%d", range.first, range.last);
                }
            }
            dataLogF("\"]\n");
        }

        for (uint32_t epsilonTargetIndex = node.epsilonTransitionTargetsStart; epsilonTargetIndex < node.epsilonTransitionTargetsEnd; ++epsilonTargetIndex) {
            uint32_t target = epsilonTransitionsTargets[epsilonTargetIndex];
            dataLogF("        %d -> %d [label=\"É›\"]\n", i, target);
        }
    }

    dataLogF("    }\n");
    dataLogF("}\n");
}
#endif

} // namespace ContentExtensions

} // namespace WebCore

#endif // ENABLE(CONTENT_EXTENSIONS)
