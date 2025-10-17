/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 4, 2024.
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

#include <algorithm>
#include <cstdint>

#if ENABLE(DFG_JIT)

namespace JSC { namespace DFG {

enum class AbstractInterpreterClobberState : uint8_t {
    NotClobbered,
    FoldedClobber,
    ObservedTransitions,
    ClobberedStructures
};

inline AbstractInterpreterClobberState mergeClobberStates(AbstractInterpreterClobberState a, AbstractInterpreterClobberState b)
{
    uint8_t aInt = static_cast<uint8_t>(a);
    uint8_t bInt = static_cast<uint8_t>(b);
    return static_cast<AbstractInterpreterClobberState>(std::max(aInt, bInt));
}

} } // namespace JSC::DFG

namespace WTF {

class PrintStream;

void printInternal(PrintStream&, JSC::DFG::AbstractInterpreterClobberState);

} // namespace WTF

#endif // ENABLE(DFG_JIT)

