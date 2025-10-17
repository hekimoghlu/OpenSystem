/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 25, 2021.
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

#if ENABLE(DFG_JIT)

#include "DFGInlineCacheWrapper.h"
#include "DFGSlowPathGenerator.h"

namespace JSC { namespace DFG {

template<typename GeneratorType>
void InlineCacheWrapper<GeneratorType>::finalize(LinkBuffer& fastPath, LinkBuffer& slowPath)
{
    m_generator.reportSlowPathCall(m_slowPath->label(), m_slowPath->call());
    if (m_generator.m_unlinkedStubInfo) {
        m_generator.m_unlinkedStubInfo->doneLocation = fastPath.locationOf<JSInternalPtrTag>(m_generator.m_done);
        static_cast<DFG::UnlinkedStructureStubInfo*>(m_generator.m_unlinkedStubInfo)->slowPathStartLocation = fastPath.locationOf<JITStubRoutinePtrTag>(m_generator.m_slowPathBegin);
    } else
        m_generator.finalize(fastPath, slowPath);
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
