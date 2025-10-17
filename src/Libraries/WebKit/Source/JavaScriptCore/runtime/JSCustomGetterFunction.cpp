/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 17, 2022.
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
#include "JSCustomGetterFunction.h"

#include "IdentifierInlines.h"
#include "JSCJSValueInlines.h"
#include <wtf/text/MakeString.h>

namespace JSC {

const ClassInfo JSCustomGetterFunction::s_info = { "Function"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSCustomGetterFunction) };
static JSC_DECLARE_HOST_FUNCTION(customGetterFunctionCall);

JSC_DEFINE_HOST_FUNCTION(customGetterFunctionCall, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSCustomGetterFunction* customGetterFunction = jsCast<JSCustomGetterFunction*>(callFrame->jsCallee());
    JSValue thisValue = callFrame->thisValue();
    auto getter = customGetterFunction->getter();

    if (auto domAttribute = customGetterFunction->domAttribute()) {
        if (!thisValue.inherits(domAttribute->classInfo))
            return throwVMDOMAttributeGetterTypeError(globalObject, scope, domAttribute->classInfo, customGetterFunction->propertyName());
    }

    RELEASE_AND_RETURN(scope, getter(globalObject, JSValue::encode(thisValue), customGetterFunction->propertyName()));
}

JSCustomGetterFunction::JSCustomGetterFunction(VM& vm, NativeExecutable* executable, JSGlobalObject* globalObject, Structure* structure, const PropertyName& propertyName, CustomFunctionPointer getter, std::optional<DOMAttributeAnnotation> domAttribute)
    : Base(vm, executable, globalObject, structure)
    , m_propertyName(Identifier::fromUid(vm, propertyName.uid()))
    , m_getter(getter)
    , m_domAttribute(domAttribute)
{
}

JSCustomGetterFunction* JSCustomGetterFunction::create(VM& vm, JSGlobalObject* globalObject, const PropertyName& propertyName, CustomFunctionPointer getter, std::optional<DOMAttributeAnnotation> domAttribute)
{
    ASSERT(getter);
    NativeExecutable* executable = vm.getHostFunction(customGetterFunctionCall, ImplementationVisibility::Public, callHostFunctionAsConstructor, String(propertyName.publicName()));
    Structure* structure = globalObject->customGetterFunctionStructure();
    JSCustomGetterFunction* function = new (NotNull, allocateCell<JSCustomGetterFunction>(vm)) JSCustomGetterFunction(vm, executable, globalObject, structure, propertyName, getter, domAttribute);

    // Can't do this during initialization because getHostFunction might do a GC allocation.
    auto name = makeString("get "_s, propertyName.publicName());
    function->finishCreation(vm, executable, 0, name);
    return function;
}

void JSCustomGetterFunction::destroy(JSCell* cell)
{
    static_cast<JSCustomGetterFunction*>(cell)->~JSCustomGetterFunction();
}

} // namespace JSC
