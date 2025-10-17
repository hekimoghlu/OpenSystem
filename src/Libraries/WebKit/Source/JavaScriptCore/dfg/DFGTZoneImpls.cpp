/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 29, 2024.
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

#if ENABLE(DFG_JIT)

#include "DFGAbstractInterpreter.h"
#include "DFGAbstractValue.h"
#include "DFGArrayifySlowPathGenerator.h"
#include "DFGBackwardsDominators.h"
#include "DFGCFG.h"
#include "DFGCallArrayAllocatorSlowPathGenerator.h"
#include "DFGCallCreateDirectArgumentsSlowPathGenerator.h"
#include "DFGControlEquivalenceAnalysis.h"
#include "DFGDominators.h"
#include "DFGFlowMap.h"
#include "DFGInPlaceAbstractState.h"
#include "DFGNaturalLoops.h"
#include "DFGSaneStringGetByValSlowPathGenerator.h"
#include "DFGSlowPathGenerator.h"
#include <wtf/TZoneMallocInlines.h>

namespace JSC { namespace DFG {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ArrayifySlowPathGenerator);
WTF_MAKE_TZONE_ALLOCATED_IMPL(BackwardsCFG);
WTF_MAKE_TZONE_ALLOCATED_IMPL(BackwardsDominators);
WTF_MAKE_TZONE_ALLOCATED_IMPL(CFG);
WTF_MAKE_TZONE_ALLOCATED_IMPL(CallArrayAllocatorSlowPathGenerator);
WTF_MAKE_TZONE_ALLOCATED_IMPL(CallArrayAllocatorWithVariableSizeSlowPathGenerator);
WTF_MAKE_TZONE_ALLOCATED_IMPL(CallArrayAllocatorWithVariableStructureVariableSizeSlowPathGenerator);
WTF_MAKE_TZONE_ALLOCATED_IMPL(CallCreateDirectArgumentsSlowPathGenerator);
WTF_MAKE_TZONE_ALLOCATED_IMPL(ControlEquivalenceAnalysis);
WTF_MAKE_TZONE_ALLOCATED_IMPL(SaneStringGetByValSlowPathGenerator);

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
