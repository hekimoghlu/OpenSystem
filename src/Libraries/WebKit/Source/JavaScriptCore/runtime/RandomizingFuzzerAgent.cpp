/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 16, 2021.
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
#include "RandomizingFuzzerAgent.h"

#include "CodeBlock.h"
#include <wtf/Locker.h>
#include <wtf/TZoneMallocInlines.h>

namespace JSC {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RandomizingFuzzerAgent);

RandomizingFuzzerAgent::RandomizingFuzzerAgent(VM&)
    : m_random(Options::seedOfRandomizingFuzzerAgent())
{
}

SpeculatedType RandomizingFuzzerAgent::getPrediction(CodeBlock* codeBlock, const CodeOrigin& codeOrigin, SpeculatedType original)
{
    Locker locker { m_lock };
    uint32_t high = m_random.getUint32();
    uint32_t low = m_random.getUint32();
    SpeculatedType generated = static_cast<SpeculatedType>((static_cast<uint64_t>(high) << 32) | low) & SpecFullTop;
    if (Options::dumpFuzzerAgentPredictions())
        dataLogLn("getPrediction name:(", codeBlock->inferredName(), "#", codeBlock->hashAsStringIfPossible(), "),bytecodeIndex:(", codeOrigin.bytecodeIndex(), "),original:(", SpeculationDump(original), "),generated:(", SpeculationDump(generated), ")");
    return generated;
}

} // namespace JSC
