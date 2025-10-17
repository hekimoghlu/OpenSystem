/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 4, 2022.
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
#include "ErrorPrototype.h"

#include "IntegrityInlines.h"
#include "JSCInlines.h"
#include "StringRecursionChecker.h"

namespace JSC {

STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(ErrorPrototypeBase);
STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(ErrorPrototype);

static JSC_DECLARE_HOST_FUNCTION(errorProtoFuncToString);

}

#include "ErrorPrototype.lut.h"

namespace JSC {

const ClassInfo ErrorPrototype::s_info = { "Object"_s, &Base::s_info, &errorPrototypeTable, nullptr, CREATE_METHOD_TABLE(ErrorPrototype) };

/* Source for ErrorPrototype.lut.h
@begin errorPrototypeTable
  toString          errorProtoFuncToString         DontEnum|Function 0
@end
*/

ErrorPrototypeBase::ErrorPrototypeBase(VM& vm, Structure* structure)
    : Base(vm, structure)
{
}

void ErrorPrototypeBase::finishCreation(VM& vm, const String& name)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));
    putDirectWithoutTransition(vm, vm.propertyNames->name, jsString(vm, name), static_cast<unsigned>(PropertyAttribute::DontEnum));
    putDirectWithoutTransition(vm, vm.propertyNames->message, jsEmptyString(vm), static_cast<unsigned>(PropertyAttribute::DontEnum));
}

ErrorPrototype::ErrorPrototype(VM& vm, Structure* structure)
    : Base(vm, structure)
{
}

// ------------------------------ Functions ---------------------------

// ECMA-262 5.1, 15.11.4.4
JSC_DEFINE_HOST_FUNCTION(errorProtoFuncToString, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    // 1. Let O be the this value.
    JSValue thisValue = callFrame->thisValue().toThis(globalObject, ECMAMode::strict());

    // 2. If Type(O) is not Object, throw a TypeError exception.
    if (!thisValue.isObject())
        return throwVMTypeError(globalObject, scope);
    JSObject* thisObj = asObject(thisValue);
    Integrity::auditStructureID(thisObj->structureID());

    // Guard against recursion!
    StringRecursionChecker checker(globalObject, thisObj);
    EXCEPTION_ASSERT(!scope.exception() || checker.earlyReturnValue());
    if (JSValue earlyReturnValue = checker.earlyReturnValue())
        return JSValue::encode(earlyReturnValue);

    // 3. Let name be the result of calling the [[Get]] internal method of O with argument "name".
    JSValue name = thisObj->get(globalObject, vm.propertyNames->name);
    RETURN_IF_EXCEPTION(scope, encodedJSValue());

    // 4. If name is undefined, then let name be "Error"; else let name be ToString(name).
    String nameString;
    if (name.isUndefined())
        nameString = "Error"_s;
    else {
        nameString = name.toWTFString(globalObject);
        RETURN_IF_EXCEPTION(scope, encodedJSValue());
    }

    // 5. Let msg be the result of calling the [[Get]] internal method of O with argument "message".
    JSValue message = thisObj->get(globalObject, vm.propertyNames->message);
    RETURN_IF_EXCEPTION(scope, encodedJSValue());

    // (sic)
    // 6. If msg is undefined, then let msg be the empty String; else let msg be ToString(msg).
    // 7. If msg is undefined, then let msg be the empty String; else let msg be ToString(msg).
    String messageString;
    if (message.isUndefined())
        messageString = String();
    else {
        messageString = message.toWTFString(globalObject);
        RETURN_IF_EXCEPTION(scope, encodedJSValue());
    }

    // 8. If name is the empty String, return msg.
    if (!nameString.length())
        return JSValue::encode(message.isString() ? message : jsString(vm, WTFMove(messageString)));

    // 9. If msg is the empty String, return name.
    if (!messageString.length())
        return JSValue::encode(name.isString() ? name : jsString(vm, WTFMove(nameString)));

    // 10. Return the result of concatenating name, ":", a single space character, and msg.
    RELEASE_AND_RETURN(scope, JSValue::encode(jsMakeNontrivialString(globalObject, nameString, ": "_s, messageString)));
}

} // namespace JSC
