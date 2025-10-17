/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 7, 2022.
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
#include "SamplingCounter.h"

namespace JSC {

void AbstractSamplingCounter::dump()
{
#if ENABLE(SAMPLING_COUNTERS)
    if (s_abstractSamplingCounterChain != &s_abstractSamplingCounterChainEnd) {
        dataLogF("\nSampling Counter Values:\n");
        for (AbstractSamplingCounter* currCounter = s_abstractSamplingCounterChain; (currCounter != &s_abstractSamplingCounterChainEnd); currCounter = currCounter->m_next)
            dataLogF("\t%s\t: %lld\n", currCounter->m_name, currCounter->m_counter);
        dataLogF("\n\n");
    }
    s_completed = true;
#endif
}

AbstractSamplingCounter AbstractSamplingCounter::s_abstractSamplingCounterChainEnd;
AbstractSamplingCounter* AbstractSamplingCounter::s_abstractSamplingCounterChain = &s_abstractSamplingCounterChainEnd;
bool AbstractSamplingCounter::s_completed = false;

} // namespace JSC

