/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 30, 2025.
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
#include "WasmStreamingPlan.h"

#if ENABLE(WEBASSEMBLY)

#include "WasmCallee.h"
#include "WasmEntryPlan.h"
#include "WasmNameSection.h"
#include "WasmTypeDefinitionInlines.h"
#include <wtf/DataLog.h>
#include <wtf/Locker.h>
#include <wtf/StdLibExtras.h>

namespace JSC { namespace Wasm {

namespace WasmStreamingPlanInternal {
static constexpr bool verbose = false;
}

StreamingPlan::StreamingPlan(VM& vm, Ref<ModuleInformation>&& info, Ref<EntryPlan>&& plan, FunctionCodeIndex functionIndex, CompletionTask&& task)
    : Base(vm, WTFMove(info), WTFMove(task))
    , m_plan(WTFMove(plan))
    , m_functionIndex(functionIndex)
{
    dataLogLnIf(WasmStreamingPlanInternal::verbose, "Starting Streaming plan for ", functionIndex, " of module info: ", RawPointer(&m_moduleInformation.get()));
}

void StreamingPlan::work(CompilationEffort)
{
    m_plan->compileFunction(m_functionIndex);
    dataLogLnIf(WasmStreamingPlanInternal::verbose, "Finished Streaming ", m_functionIndex);
    Locker locker { m_lock };
    complete();
}

} } // namespace JSC::Wasm

#endif // ENABLE(WEBASSEMBLY)
