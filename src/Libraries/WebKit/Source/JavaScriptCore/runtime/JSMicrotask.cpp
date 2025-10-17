/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 16, 2022.
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
#include "JSMicrotask.h"

#include "CatchScope.h"
#include "Debugger.h"
#include "DeferTermination.h"
#include "JSGlobalObject.h"
#include "JSObjectInlines.h"
#include "Microtask.h"

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

void runJSMicrotask(JSGlobalObject* globalObject, MicrotaskIdentifier identifier, JSValue job, std::span<const JSValue> arguments)
{
    VM& vm = globalObject->vm();

    auto scope = DECLARE_CATCH_SCOPE(vm);

    if (UNLIKELY(!job.isObject()))
        return;

    // If termination is issued, do not run microtasks. Otherwise, microtask should not care about exceptions.
    if (UNLIKELY(!scope.clearExceptionExceptTermination()))
        return;

    auto handlerCallData = JSC::getCallData(job);
    if (UNLIKELY(!scope.clearExceptionExceptTermination()))
        return;
    ASSERT(handlerCallData.type != CallData::Type::None);

    unsigned count = 0;
    for (auto argument : arguments) {
        if (!argument)
            break;
        ++count;
    }

    if (UNLIKELY(globalObject->hasDebugger())) {
        DeferTerminationForAWhile deferTerminationForAWhile(vm);
        globalObject->debugger()->willRunMicrotask(globalObject, identifier);
        scope.clearException();
    }

    if (LIKELY(!vm.hasPendingTerminationException())) {
        profiledCall(globalObject, ProfilingReason::Microtask, job, handlerCallData, jsUndefined(), ArgList { std::bit_cast<EncodedJSValue*>(arguments.data()), count });
        scope.clearExceptionExceptTermination();
    }

    if (UNLIKELY(globalObject->hasDebugger())) {
        DeferTerminationForAWhile deferTerminationForAWhile(vm);
        globalObject->debugger()->didRunMicrotask(globalObject, identifier);
        scope.clearException();
    }
}

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
