/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 14, 2024.
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
#include "NarrowingNumberPredictionFuzzerAgent.h"

#include "CodeBlock.h"
#include <wtf/TZoneMallocInlines.h>

namespace JSC {

WTF_MAKE_TZONE_ALLOCATED_IMPL(NarrowingNumberPredictionFuzzerAgent);

NarrowingNumberPredictionFuzzerAgent::NarrowingNumberPredictionFuzzerAgent(VM& vm)
    : NumberPredictionFuzzerAgent(vm)
{
}

SpeculatedType NarrowingNumberPredictionFuzzerAgent::getPrediction(CodeBlock* codeBlock, const CodeOrigin& codeOrigin, SpeculatedType original)
{
    Locker locker { m_lock };

    if (!(original && speculationChecked(original, SpecBytecodeNumber)))
        return original;

    Vector<SpeculatedType> numberTypesThatCouldBePartOfSpeculation;
    for (auto numberType : bytecodeNumberTypes()) {
        if (numberType & original)
            numberTypesThatCouldBePartOfSpeculation.append(numberType);
    }

    unsigned numberOfTypesToKeep = m_random.getUint32(numberTypesThatCouldBePartOfSpeculation.size()) + 1;
    if (numberOfTypesToKeep == numberTypesThatCouldBePartOfSpeculation.size())
        return original;

    SpeculatedType generated = SpecNone;
    for (unsigned i = 0; i < numberOfTypesToKeep; i++) {
        unsigned indexOfTypeToKeep = m_random.getUint32(numberTypesThatCouldBePartOfSpeculation.size());
        mergeSpeculation(generated, numberTypesThatCouldBePartOfSpeculation[indexOfTypeToKeep]);
        numberTypesThatCouldBePartOfSpeculation.remove(indexOfTypeToKeep);
    }

    if (Options::dumpFuzzerAgentPredictions())
        dataLogLn("NarrowingNumberPredictionFuzzerAgent::getPrediction name:(", codeBlock->inferredName(), "#", codeBlock->hashAsStringIfPossible(), "),bytecodeIndex:(", codeOrigin.bytecodeIndex(), "),original:(", SpeculationDump(original), "),generated:(", SpeculationDump(generated), ")");

    return generated;
}

} // namespace JSC
