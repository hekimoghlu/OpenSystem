/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 5, 2023.
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
#include "JSCustomSetterFunction.h"

#include "IdentifierInlines.h"
#include "JSCJSValueInlines.h"
#include <wtf/text/MakeString.h>

namespace JSC {

const ClassInfo JSCustomSetterFunction::s_info = { "Function"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSCustomSetterFunction) };
static JSC_DECLARE_HOST_FUNCTION(customSetterFunctionCall);

JSC_DEFINE_HOST_FUNCTION(customSetterFunctionCall, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    auto customSetterFunction = jsCast<JSCustomSetterFunction*>(callFrame->jsCallee());
    auto setter = customSetterFunction->setter();
    setter(globalObject, JSValue::encode(callFrame->thisValue()), JSValue::encode(callFrame->argument(0)), customSetterFunction->propertyName());
    return JSValue::encode(jsUndefined());
}

JSCustomSetterFunction::JSCustomSetterFunction(VM& vm, NativeExecutable* executable, JSGlobalObject* globalObject, Structure* structure, const PropertyName& propertyName, CustomFunctionPointer setter)
    : Base(vm, executable, globalObject, structure)
    , m_propertyName(Identifier::fromUid(vm, propertyName.uid()))
    , m_setter(setter)
{
}

JSCustomSetterFunction* JSCustomSetterFunction::create(VM& vm, JSGlobalObject* globalObject, const PropertyName& propertyName, CustomFunctionPointer setter)
{
    ASSERT(setter);
    NativeExecutable* executable = vm.getHostFunction(customSetterFunctionCall, ImplementationVisibility::Public, callHostFunctionAsConstructor, String(propertyName.publicName()));
    Structure* structure = globalObject->customSetterFunctionStructure();
    JSCustomSetterFunction* function = new (NotNull, allocateCell<JSCustomSetterFunction>(vm)) JSCustomSetterFunction(vm, executable, globalObject, structure, propertyName, setter);

    // Can't do this during initialization because getHostFunction might do a GC allocation.
    auto name = makeString("set "_s, propertyName.publicName());
    function->finishCreation(vm, executable, 1, name);
    return function;
}

void JSCustomSetterFunction::destroy(JSCell* cell)
{
    static_cast<JSCustomSetterFunction*>(cell)->~JSCustomSetterFunction();
}

} // namespace JSC
