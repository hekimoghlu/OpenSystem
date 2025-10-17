/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 8, 2023.
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
#include "ConstructData.h"

#include "Interpreter.h"
#include "JSGlobalObject.h"
#include "JSObjectInlines.h"
#include "ScriptProfilingScope.h"

namespace JSC {

JSObject* construct(JSGlobalObject* globalObject, JSValue constructorObject, const ArgList& args, ASCIILiteral errorMessage)
{
    return construct(globalObject, constructorObject, constructorObject, args, errorMessage);
}

JSObject* construct(JSGlobalObject* globalObject, JSValue constructorObject, JSValue newTarget, const ArgList& args, ASCIILiteral errorMessage)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto constructData = JSC::getConstructData(constructorObject);
    if (UNLIKELY(constructData.type == CallData::Type::None)) {
        throwTypeError(globalObject, scope, errorMessage);
        return nullptr;
    }

    RELEASE_AND_RETURN(scope, construct(globalObject, constructorObject, constructData, args, newTarget));
}

JSObject* construct(JSGlobalObject* globalObject, JSValue constructorObject, const CallData& constructData, const ArgList& args, JSValue newTarget)
{
    VM& vm = globalObject->vm();
    ASSERT(constructData.type == CallData::Type::JS || constructData.type == CallData::Type::Native);
    return vm.interpreter.executeConstruct(asObject(constructorObject), constructData, args, newTarget);
}

JSObject* profiledConstruct(JSGlobalObject* globalObject, ProfilingReason reason, JSValue constructorObject, const CallData& constructData, const ArgList& args, JSValue newTarget)
{
    VM& vm = globalObject->vm();
    ScriptProfilingScope profilingScope(vm.deprecatedVMEntryGlobalObject(globalObject), reason);
    return construct(globalObject, constructorObject, constructData, args, newTarget);
}

} // namespace JSC
