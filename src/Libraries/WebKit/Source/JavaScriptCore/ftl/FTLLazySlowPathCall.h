/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 24, 2025.
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

#if ENABLE(FTL_JIT)

#include "CodeBlock.h"
#include "CodeLocation.h"
#include "FTLLazySlowPath.h"
#include "FTLSlowPathCall.h"
#include "FTLThunks.h"
#include "GPRInfo.h"
#include "MacroAssemblerCodeRef.h"
#include "RegisterSet.h"

namespace JSC { namespace FTL {

template<typename ResultType, typename... ArgumentTypes>
Ref<LazySlowPath::Generator> createLazyCallGenerator(
    VM& vm, CodePtr<CFunctionPtrTag> function, ResultType result, ArgumentTypes... arguments)
{
    return LazySlowPath::createGenerator(
        [=, &vm] (CCallHelpers& jit, LazySlowPath::GenerationParams& params) {
            callOperation(
                vm, params.lazySlowPath->usedRegisters(), jit, params.lazySlowPath->callSiteIndex(),
                params.exceptionJumps, function, result, arguments...);
            params.doneJumps.append(jit.jump());
        });
}

} } // namespace JSC::FTL

#endif // ENABLE(FTL_JIT)
