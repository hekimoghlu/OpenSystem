/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 29, 2023.
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
#include "JSPromisePrototype.h"

#include "BuiltinNames.h"
#include "JSCInlines.h"

namespace JSC {

STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(JSPromisePrototype);

}

#include "JSPromisePrototype.lut.h"

namespace JSC {

const ClassInfo JSPromisePrototype::s_info = { "Promise"_s, &Base::s_info, &promisePrototypeTable, nullptr, CREATE_METHOD_TABLE(JSPromisePrototype) };

/* Source for JSPromisePrototype.lut.h
@begin promisePrototypeTable
  catch        JSBuiltin            DontEnum|Function 1
  finally      JSBuiltin            DontEnum|Function 1
@end
*/

JSPromisePrototype* JSPromisePrototype::create(VM& vm, JSGlobalObject* globalObject, Structure* structure)
{
    JSPromisePrototype* object = new (NotNull, allocateCell<JSPromisePrototype>(vm)) JSPromisePrototype(vm, structure);
    object->finishCreation(vm, globalObject);
    object->addOwnInternalSlots(vm, globalObject);
    return object;
}

Structure* JSPromisePrototype::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(ObjectType, StructureFlags), info());
}

JSPromisePrototype::JSPromisePrototype(VM& vm, Structure* structure)
    : JSNonFinalObject(vm, structure)
{
}

void JSPromisePrototype::finishCreation(VM& vm, JSGlobalObject* globalObject)
{
    Base::finishCreation(vm);
    putDirectWithoutTransition(vm, vm.propertyNames->builtinNames().thenPublicName(), globalObject->promiseProtoThenFunction(), static_cast<unsigned>(PropertyAttribute::DontEnum));
    JSC_TO_STRING_TAG_WITHOUT_TRANSITION();
}

void JSPromisePrototype::addOwnInternalSlots(VM& vm, JSGlobalObject* globalObject)
{
    putDirectWithoutTransition(vm, vm.propertyNames->builtinNames().thenPrivateName(), globalObject->promiseProtoThenFunction(), PropertyAttribute::DontEnum | PropertyAttribute::DontDelete | PropertyAttribute::ReadOnly);
}

} // namespace JSC
