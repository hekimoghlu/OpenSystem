/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 26, 2023.
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
#include "WasmTierUpCount.h"

#if ENABLE(WEBASSEMBLY_OMGJIT) || ENABLE(WEBASSEMBLY_BBQJIT)

#include "WasmOSREntryData.h"
#include <wtf/TZoneMallocInlines.h>

namespace JSC { namespace Wasm {

WTF_MAKE_TZONE_ALLOCATED_IMPL(TierUpCount);

TierUpCount::TierUpCount()
{
    setNewThreshold(Options::thresholdForOMGOptimizeAfterWarmUp());
    m_compilationStatusForOMG.fill(CompilationStatus::NotCompiled);
    m_compilationStatusForOMGForOSREntry.fill(CompilationStatus::NotCompiled);
}

TierUpCount::~TierUpCount() = default;

OSREntryData& TierUpCount::addOSREntryData(FunctionCodeIndex functionIndex, uint32_t loopIndex, StackMap&& stackMap)
{
    m_osrEntryData.append(makeUnique<OSREntryData>(functionIndex, loopIndex, WTFMove(stackMap)));
    return *m_osrEntryData.last().get();
}

OSREntryData& TierUpCount::osrEntryData(uint32_t loopIndex)
{
    return *m_osrEntryData[loopIndex];
}

} } // namespace JSC::Wasm

#endif // ENABLE(WEBASSEMBLY_OMGJIT) || ENABLE(WEBASSEMBLY_BBQJIT)
