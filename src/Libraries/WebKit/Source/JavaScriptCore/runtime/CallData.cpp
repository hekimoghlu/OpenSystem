/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 9, 2024.
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
#include "CallData.h"

#include "CatchScope.h"
#include "Interpreter.h"
#include "JSObjectInlines.h"
#include "ScriptProfilingScope.h"

namespace JSC {

JSValue call(JSGlobalObject* globalObject, JSValue functionObject, const ArgList& args, ASCIILiteral errorMessage)
{
    return call(globalObject, functionObject, functionObject, args, errorMessage);
}

JSValue call(JSGlobalObject* globalObject, JSValue functionObject, JSValue thisValue, const ArgList& args, ASCIILiteral errorMessage)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto callData = JSC::getCallData(functionObject);
    if (callData.type == CallData::Type::None)
        return throwTypeError(globalObject, scope, errorMessage);

    RELEASE_AND_RETURN(scope, call(globalObject, functionObject, callData, thisValue, args));
}

JSValue call(JSGlobalObject* globalObject, JSValue functionObject, const CallData& callData, JSValue thisValue, const ArgList& args)
{
    VM& vm = globalObject->vm();
    ASSERT(callData.type == CallData::Type::JS || callData.type == CallData::Type::Native);
    return vm.interpreter.executeCall(asObject(functionObject), callData, thisValue, args);
}

JSValue call(JSGlobalObject* globalObject, JSValue functionObject, const CallData& callData, JSValue thisValue, const ArgList& args, NakedPtr<Exception>& returnedException)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_CATCH_SCOPE(vm);
    JSValue result = call(globalObject, functionObject, callData, thisValue, args);
    if (UNLIKELY(scope.exception())) {
        returnedException = scope.exception();
        scope.clearException();
        return jsUndefined();
    }
    RELEASE_ASSERT(result);
    return result;
}

JSValue profiledCall(JSGlobalObject* globalObject, ProfilingReason reason, JSValue functionObject, const CallData& callData, JSValue thisValue, const ArgList& args)
{
    VM& vm = globalObject->vm();
    ScriptProfilingScope profilingScope(vm.deprecatedVMEntryGlobalObject(globalObject), reason);
    return call(globalObject, functionObject, callData, thisValue, args);
}

JSValue profiledCall(JSGlobalObject* globalObject, ProfilingReason reason, JSValue functionObject, const CallData& callData, JSValue thisValue, const ArgList& args, NakedPtr<Exception>& returnedException)
{
    VM& vm = globalObject->vm();
    ScriptProfilingScope profilingScope(vm.deprecatedVMEntryGlobalObject(globalObject), reason);
    return call(globalObject, functionObject, callData, thisValue, args, returnedException);
}

} // namespace JSC
