/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 16, 2025.
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
#include "ReflectObject.h"

#include "JSCInlines.h"
#include "ObjectConstructor.h"

namespace JSC {

static JSC_DECLARE_HOST_FUNCTION(reflectObjectConstruct);
static JSC_DECLARE_HOST_FUNCTION(reflectObjectDefineProperty);
static JSC_DECLARE_HOST_FUNCTION(reflectObjectGetOwnPropertyDescriptor);
static JSC_DECLARE_HOST_FUNCTION(reflectObjectGetPrototypeOf);
static JSC_DECLARE_HOST_FUNCTION(reflectObjectIsExtensible);
static JSC_DECLARE_HOST_FUNCTION(reflectObjectOwnKeys);
static JSC_DECLARE_HOST_FUNCTION(reflectObjectPreventExtensions);
static JSC_DECLARE_HOST_FUNCTION(reflectObjectSet);
static JSC_DECLARE_HOST_FUNCTION(reflectObjectSetPrototypeOf);

}

#include "ReflectObject.lut.h"

namespace JSC {

const ASCIILiteral ReflectOwnKeysNonObjectArgumentError { "Reflect.ownKeys requires the first argument be an object"_s };

STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(ReflectObject);

const ClassInfo ReflectObject::s_info = { "Reflect"_s, &Base::s_info, &reflectObjectTable, nullptr, CREATE_METHOD_TABLE(ReflectObject) };

/* Source for ReflectObject.lut.h
@begin reflectObjectTable
    apply                    JSBuiltin                             DontEnum|Function 3
    construct                reflectObjectConstruct                DontEnum|Function 2
    defineProperty           reflectObjectDefineProperty           DontEnum|Function 3
    deleteProperty           JSBuiltin                             DontEnum|Function 2
    get                      JSBuiltin                             DontEnum|Function 2
    getOwnPropertyDescriptor reflectObjectGetOwnPropertyDescriptor DontEnum|Function 2
    getPrototypeOf           reflectObjectGetPrototypeOf           DontEnum|Function 1 ReflectGetPrototypeOfIntrinsic
    has                      JSBuiltin                             DontEnum|Function 2
    isExtensible             reflectObjectIsExtensible             DontEnum|Function 1
    ownKeys                  reflectObjectOwnKeys                  DontEnum|Function 1 ReflectOwnKeysIntrinsic
    preventExtensions        reflectObjectPreventExtensions        DontEnum|Function 1
    set                      reflectObjectSet                      DontEnum|Function 3
    setPrototypeOf           reflectObjectSetPrototypeOf           DontEnum|Function 2
@end
*/

ReflectObject::ReflectObject(VM& vm, Structure* structure)
    : JSNonFinalObject(vm, structure)
{
}

void ReflectObject::finishCreation(VM& vm, JSGlobalObject*)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));
    JSC_TO_STRING_TAG_WITHOUT_TRANSITION();
}

// ------------------------------ Functions --------------------------------

// https://tc39.github.io/ecma262/#sec-reflect.construct
JSC_DEFINE_HOST_FUNCTION(reflectObjectConstruct, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue target = callFrame->argument(0);
    if (!target.isObject())
        return JSValue::encode(throwTypeError(globalObject, scope, "Reflect.construct requires the first argument be a constructor"_s));

    auto constructData = JSC::getConstructData(target);
    if (constructData.type == CallData::Type::None)
        return JSValue::encode(throwTypeError(globalObject, scope, "Reflect.construct requires the first argument be a constructor"_s));

    JSValue newTarget = target;
    if (callFrame->argumentCount() >= 3) {
        newTarget = callFrame->argument(2);
        if (!newTarget.isConstructor())
            return JSValue::encode(throwTypeError(globalObject, scope, "Reflect.construct requires the third argument be a constructor if present"_s));
    }

    MarkedArgumentBuffer arguments;
    JSObject* argumentsObject = jsDynamicCast<JSObject*>(callFrame->argument(1));
    if (!argumentsObject)
        return JSValue::encode(throwTypeError(globalObject, scope, "Reflect.construct requires the second argument be an object"_s));

    forEachInArrayLike(globalObject, argumentsObject, [&] (JSValue value) -> bool {
        arguments.append(value);
        return true;
    });
    RETURN_IF_EXCEPTION(scope, (arguments.overflowCheckNotNeeded(), encodedJSValue()));
    if (UNLIKELY(arguments.hasOverflowed())) {
        throwOutOfMemoryError(globalObject, scope);
        return encodedJSValue();
    }

    RELEASE_AND_RETURN(scope, JSValue::encode(construct(globalObject, target, constructData, arguments, newTarget)));
}

// https://tc39.github.io/ecma262/#sec-reflect.defineproperty
JSC_DEFINE_HOST_FUNCTION(reflectObjectDefineProperty, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue target = callFrame->argument(0);
    if (!target.isObject())
        return JSValue::encode(throwTypeError(globalObject, scope, "Reflect.defineProperty requires the first argument be an object"_s));
    auto propertyName = callFrame->argument(1).toPropertyKey(globalObject);
    RETURN_IF_EXCEPTION(scope, encodedJSValue());

    PropertyDescriptor descriptor;
    bool success = toPropertyDescriptor(globalObject, callFrame->argument(2), descriptor);
    EXCEPTION_ASSERT(!scope.exception() == success);
    if (UNLIKELY(!success))
        return encodedJSValue();
    ASSERT((descriptor.attributes() & PropertyAttribute::Accessor) || (!descriptor.isAccessorDescriptor()));
    scope.assertNoException();

    // Reflect.defineProperty should not throw an error when the defineOwnProperty operation fails.
    bool shouldThrow = false;
    JSObject* targetObject = asObject(target);
    RELEASE_AND_RETURN(scope, JSValue::encode(jsBoolean(targetObject->methodTable()->defineOwnProperty(targetObject, globalObject, propertyName, descriptor, shouldThrow))));
}

// https://tc39.github.io/ecma262/#sec-reflect.getownpropertydescriptor
JSC_DEFINE_HOST_FUNCTION(reflectObjectGetOwnPropertyDescriptor, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue target = callFrame->argument(0);
    if (!target.isObject())
        return JSValue::encode(throwTypeError(globalObject, scope, "Reflect.getOwnPropertyDescriptor requires the first argument be an object"_s));

    auto key = callFrame->argument(1).toPropertyKey(globalObject);
    RETURN_IF_EXCEPTION(scope, encodedJSValue());

    RELEASE_AND_RETURN(scope, JSValue::encode(objectConstructorGetOwnPropertyDescriptor(globalObject, asObject(target), key)));
}

// https://tc39.github.io/ecma262/#sec-reflect.getprototypeof
JSC_DEFINE_HOST_FUNCTION(reflectObjectGetPrototypeOf, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue target = callFrame->argument(0);
    if (!target.isObject())
        return JSValue::encode(throwTypeError(globalObject, scope, "Reflect.getPrototypeOf requires the first argument be an object"_s));
    RELEASE_AND_RETURN(scope, JSValue::encode(asObject(target)->getPrototype(vm, globalObject)));
}

// https://tc39.github.io/ecma262/#sec-reflect.isextensible
JSC_DEFINE_HOST_FUNCTION(reflectObjectIsExtensible, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue target = callFrame->argument(0);
    if (!target.isObject())
        return JSValue::encode(throwTypeError(globalObject, scope, "Reflect.isExtensible requires the first argument be an object"_s));

    bool isExtensible = asObject(target)->isExtensible(globalObject);
    RETURN_IF_EXCEPTION(scope, encodedJSValue());
    return JSValue::encode(jsBoolean(isExtensible));
}

// https://tc39.github.io/ecma262/#sec-reflect.ownkeys
JSC_DEFINE_HOST_FUNCTION(reflectObjectOwnKeys, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue target = callFrame->argument(0);
    if (!target.isObject())
        return JSValue::encode(throwTypeError(globalObject, scope, ReflectOwnKeysNonObjectArgumentError));
    RELEASE_AND_RETURN(scope, JSValue::encode(ownPropertyKeys(globalObject, asObject(target), PropertyNameMode::StringsAndSymbols, DontEnumPropertiesMode::Include)));
}

// https://tc39.github.io/ecma262/#sec-reflect.preventextensions
JSC_DEFINE_HOST_FUNCTION(reflectObjectPreventExtensions, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue target = callFrame->argument(0);
    if (!target.isObject())
        return JSValue::encode(throwTypeError(globalObject, scope, "Reflect.preventExtensions requires the first argument be an object"_s));
    JSObject* object = asObject(target);
    bool result = object->methodTable()->preventExtensions(object, globalObject);
    RETURN_IF_EXCEPTION(scope, encodedJSValue());
    return JSValue::encode(jsBoolean(result));
}

// https://tc39.github.io/ecma262/#sec-reflect.set
JSC_DEFINE_HOST_FUNCTION(reflectObjectSet, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue target = callFrame->argument(0);
    if (!target.isObject())
        return JSValue::encode(throwTypeError(globalObject, scope, "Reflect.set requires the first argument be an object"_s));
    JSObject* targetObject = asObject(target);

    auto propertyName = callFrame->argument(1).toPropertyKey(globalObject);
    RETURN_IF_EXCEPTION(scope, encodedJSValue());

    JSValue receiver = target;
    if (callFrame->argumentCount() >= 4)
        receiver = callFrame->argument(3);

    // Do not raise any readonly errors that happen in strict mode.
    bool shouldThrowIfCantSet = false;
    PutPropertySlot slot(receiver, shouldThrowIfCantSet);
    RELEASE_AND_RETURN(scope, JSValue::encode(jsBoolean(targetObject->methodTable()->put(targetObject, globalObject, propertyName, callFrame->argument(2), slot))));
}

// https://tc39.github.io/ecma262/#sec-reflect.setprototypeof
JSC_DEFINE_HOST_FUNCTION(reflectObjectSetPrototypeOf, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue target = callFrame->argument(0);
    if (!target.isObject())
        return JSValue::encode(throwTypeError(globalObject, scope, "Reflect.setPrototypeOf requires the first argument be an object"_s));
    JSValue proto = callFrame->argument(1);
    if (!proto.isObject() && !proto.isNull())
        return JSValue::encode(throwTypeError(globalObject, scope, "Reflect.setPrototypeOf requires the second argument be either an object or null"_s));

    JSObject* object = asObject(target);

    bool shouldThrowIfCantSet = false;
    bool didSetPrototype = object->setPrototype(vm, globalObject, proto, shouldThrowIfCantSet);
    RETURN_IF_EXCEPTION(scope, encodedJSValue());
    return JSValue::encode(jsBoolean(didSetPrototype));
}

} // namespace JSC
