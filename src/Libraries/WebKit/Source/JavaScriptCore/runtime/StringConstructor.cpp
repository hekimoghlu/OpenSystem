/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 3, 2025.
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
#include "StringConstructor.h"

#include "JSCInlines.h"
#include "StringPrototype.h"
#include <wtf/text/StringBuilder.h>

namespace JSC {

static JSC_DECLARE_HOST_FUNCTION(stringFromCharCode);
static JSC_DECLARE_HOST_FUNCTION(stringFromCodePoint);

}

#include "StringConstructor.lut.h"

namespace JSC {

const ClassInfo StringConstructor::s_info = { "Function"_s, &Base::s_info, &stringConstructorTable, nullptr, CREATE_METHOD_TABLE(StringConstructor) };

/* Source for StringConstructor.lut.h
@begin stringConstructorTable
  fromCharCode          stringFromCharCode         DontEnum|Function 1 FromCharCodeIntrinsic
  fromCodePoint         stringFromCodePoint        DontEnum|Function 1
  raw                   JSBuiltin                  DontEnum|Function 1
@end
*/

STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(StringConstructor);


static JSC_DECLARE_HOST_FUNCTION(callStringConstructor);
static JSC_DECLARE_HOST_FUNCTION(constructWithStringConstructor);

StringConstructor::StringConstructor(VM& vm, NativeExecutable* executable, JSGlobalObject* globalObject, Structure* structure)
    : Base(vm, executable, globalObject, structure)
{
}

void StringConstructor::finishCreation(VM& vm, StringPrototype* stringPrototype)
{
    Base::finishCreation(vm);
    putDirectWithoutTransition(vm, vm.propertyNames->prototype, stringPrototype, PropertyAttribute::ReadOnly | PropertyAttribute::DontEnum | PropertyAttribute::DontDelete);
    putDirectWithoutTransition(vm, vm.propertyNames->length, jsNumber(1), PropertyAttribute::DontEnum | PropertyAttribute::ReadOnly);
    putDirectWithoutTransition(vm, vm.propertyNames->name, jsString(vm, vm.propertyNames->String.string()), PropertyAttribute::ReadOnly | PropertyAttribute::DontEnum);
}

StringConstructor* StringConstructor::create(VM& vm, Structure* structure, StringPrototype* stringPrototype)
{
    JSGlobalObject* globalObject = structure->globalObject();
    NativeExecutable* executable = vm.getHostFunction(callStringConstructor, ImplementationVisibility::Public, StringConstructorIntrinsic, constructWithStringConstructor, nullptr, vm.propertyNames->String.string());
    StringConstructor* constructor = new (NotNull, allocateCell<StringConstructor>(vm)) StringConstructor(vm, executable, globalObject, structure);
    constructor->finishCreation(vm, stringPrototype);
    return constructor;
}

// ------------------------------ Functions --------------------------------

JSC_DEFINE_HOST_FUNCTION(stringFromCharCode, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    unsigned length = callFrame->argumentCount();
    if (LIKELY(length == 1)) {
        scope.release();
        UChar code = callFrame->uncheckedArgument(0).toUInt32(globalObject);
        // Not checking for an exception here is ok because jsSingleCharacterString will just fetch an unused string if there's an exception.
        return JSValue::encode(jsSingleCharacterString(vm, code));
    }

    std::span<LChar> buf8Bit;
    auto impl8Bit = StringImpl::createUninitialized(length, buf8Bit);
    for (unsigned i = 0; i < length; ++i) {
        UChar character = static_cast<UChar>(callFrame->uncheckedArgument(i).toUInt32(globalObject));
        RETURN_IF_EXCEPTION(scope, encodedJSValue());
        if (UNLIKELY(!isLatin1(character))) {
            std::span<UChar> buf16Bit;
            auto impl16Bit = StringImpl::createUninitialized(length, buf16Bit);
            StringImpl::copyCharacters(buf16Bit, buf8Bit.first(i));
            buf16Bit[i] = character;
            ++i;
            for (; i < length; ++i) {
                buf16Bit[i] = static_cast<UChar>(callFrame->uncheckedArgument(i).toUInt32(globalObject));
                RETURN_IF_EXCEPTION(scope, encodedJSValue());
            }
            RELEASE_AND_RETURN(scope, JSValue::encode(jsString(vm, WTFMove(impl16Bit))));
        }
        buf8Bit[i] = static_cast<LChar>(character);
    }
    RELEASE_AND_RETURN(scope, JSValue::encode(jsString(vm, WTFMove(impl8Bit))));
}

JSString* stringFromCharCode(JSGlobalObject* globalObject, int32_t arg)
{
    return jsSingleCharacterString(globalObject->vm(), static_cast<UChar>(arg));
}

JSC_DEFINE_HOST_FUNCTION(stringFromCodePoint, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    unsigned length = callFrame->argumentCount();
    StringBuilder builder;
    builder.reserveCapacity(length);

    for (unsigned i = 0; i < length; ++i) {
        double codePointAsDouble = callFrame->uncheckedArgument(i).toNumber(globalObject);
        RETURN_IF_EXCEPTION(scope, encodedJSValue());

        uint32_t codePoint = static_cast<uint32_t>(codePointAsDouble);

        if (codePoint != codePointAsDouble || codePoint > UCHAR_MAX_VALUE)
            return throwVMError(globalObject, scope, createRangeError(globalObject, "Arguments contain a value that is out of range of code points"_s));

        if (U_IS_BMP(codePoint))
            builder.append(static_cast<UChar>(codePoint));
        else {
            builder.append(U16_LEAD(codePoint));
            builder.append(U16_TRAIL(codePoint));
        }
    }

    RELEASE_AND_RETURN(scope, JSValue::encode(jsString(vm, builder.toString())));
}

JSC_DEFINE_HOST_FUNCTION(constructWithStringConstructor, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSObject* newTarget = asObject(callFrame->newTarget());
    Structure* structure = JSC_GET_DERIVED_STRUCTURE(vm, stringObjectStructure, newTarget, callFrame->jsCallee());
    RETURN_IF_EXCEPTION(scope, { });

    if (!callFrame->argumentCount())
        return JSValue::encode(StringObject::create(vm, structure));
    JSString* str = callFrame->uncheckedArgument(0).toString(globalObject);
    RETURN_IF_EXCEPTION(scope, encodedJSValue());
    return JSValue::encode(StringObject::create(vm, structure, str));
}

JSString* stringConstructor(JSGlobalObject* globalObject, JSValue argument)
{
    VM& vm = globalObject->vm();
    if (argument.isSymbol())
        return jsNontrivialString(vm, asSymbol(argument)->descriptiveString());
    return argument.toString(globalObject);
}

JSC_DEFINE_HOST_FUNCTION(callStringConstructor, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    if (!callFrame->argumentCount())
        return JSValue::encode(jsEmptyString(vm));
    return JSValue::encode(stringConstructor(globalObject, callFrame->uncheckedArgument(0)));
}

} // namespace JSC
