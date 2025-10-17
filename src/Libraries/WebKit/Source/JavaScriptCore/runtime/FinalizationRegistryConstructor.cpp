/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 9, 2024.
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
#include "FinalizationRegistryConstructor.h"

#include "Error.h"
#include "FinalizationRegistryPrototype.h"
#include "IteratorOperations.h"
#include "JSCInlines.h"
#include "JSFinalizationRegistry.h"
#include "JSGlobalObject.h"
#include "JSObjectInlines.h"


namespace JSC {

const ClassInfo FinalizationRegistryConstructor::s_info = { "Function"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(FinalizationRegistryConstructor) };

void FinalizationRegistryConstructor::finishCreation(VM& vm, FinalizationRegistryPrototype* prototype)
{
    Base::finishCreation(vm, 1, "FinalizationRegistry"_s, PropertyAdditionMode::WithoutStructureTransition);
    putDirectWithoutTransition(vm, vm.propertyNames->prototype, prototype, PropertyAttribute::DontEnum | PropertyAttribute::DontDelete | PropertyAttribute::ReadOnly);
}

static JSC_DECLARE_HOST_FUNCTION(callFinalizationRegistry);
static JSC_DECLARE_HOST_FUNCTION(constructFinalizationRegistry);

FinalizationRegistryConstructor::FinalizationRegistryConstructor(VM& vm, Structure* structure)
    : Base(vm, structure, callFinalizationRegistry, constructFinalizationRegistry)
{
}

JSC_DEFINE_HOST_FUNCTION(callFinalizationRegistry, (JSGlobalObject* globalObject, CallFrame*))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    return JSValue::encode(throwConstructorCannotBeCalledAsFunctionTypeError(globalObject, scope, "FinalizationRegistry"_s));
}

JSC_DEFINE_HOST_FUNCTION(constructFinalizationRegistry, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    if (!callFrame->argument(0).isCallable())
        return throwVMTypeError(globalObject, scope, "First argument to FinalizationRegistry should be a function"_s);

    JSObject* newTarget = asObject(callFrame->newTarget());
    Structure* finalizationRegistryStructure = JSC_GET_DERIVED_STRUCTURE(vm, finalizationRegistryStructure, newTarget, callFrame->jsCallee());
    RETURN_IF_EXCEPTION(scope, encodedJSValue());
    RELEASE_AND_RETURN(scope, JSValue::encode(JSFinalizationRegistry::create(vm, finalizationRegistryStructure, callFrame->uncheckedArgument(0).getObject())));
}

}


