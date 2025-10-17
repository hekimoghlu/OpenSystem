/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 4, 2021.
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
#include "JSIPCBinding.h"

#if ENABLE(IPC_TESTING_API)

#include <JavaScriptCore/JSCInlines.h>
#include <JavaScriptCore/ObjectConstructor.h>
#include <WebCore/DOMException.h>
#include <WebCore/ExceptionData.h>
#include <WebCore/FloatRect.h>
#include <WebCore/IntRect.h>
#include <WebCore/RegistrableDomain.h>
#include <wtf/URL.h>

namespace IPC {

static JSC::JSValue jsValueForDecodedStringArgumentValue(JSC::JSGlobalObject* globalObject, const String& value, ASCIILiteral type)
{
    auto& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    auto* object = JSC::constructEmptyObject(globalObject, globalObject->objectPrototype());
    RETURN_IF_EXCEPTION(scope, JSC::JSValue());
    object->putDirect(vm, JSC::Identifier::fromString(vm, "type"_s), JSC::jsNontrivialString(vm, type));
    RETURN_IF_EXCEPTION(scope, JSC::JSValue());
    object->putDirect(vm, JSC::Identifier::fromString(vm, "value"_s), value.isNull() ? JSC::jsNull() : JSC::jsString(vm, value));
    RETURN_IF_EXCEPTION(scope, JSC::JSValue());
    return object;
}

template<>
JSC::JSValue jsValueForDecodedArgumentValue(JSC::JSGlobalObject* globalObject, String&& value)
{
    return jsValueForDecodedStringArgumentValue(globalObject, value, "String"_s);
}

template<>
JSC::JSValue jsValueForDecodedArgumentValue(JSC::JSGlobalObject* globalObject, URL&& value)
{
    return jsValueForDecodedStringArgumentValue(globalObject, value.string(), "URL"_s);
}

template<>
JSC::JSValue jsValueForDecodedArgumentValue(JSC::JSGlobalObject* globalObject, WebCore::RegistrableDomain&& value)
{
    return jsValueForDecodedStringArgumentValue(globalObject, value.string(), "RegistrableDomain"_s);
}

template<typename NumericType>
JSC::JSValue jsValueForDecodedNumericArgumentValue(JSC::JSGlobalObject* globalObject, NumericType value, const String& type)
{
    auto& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    auto* object = JSC::constructEmptyObject(globalObject, globalObject->objectPrototype());
    RETURN_IF_EXCEPTION(scope, JSC::JSValue());
    object->putDirect(vm, JSC::Identifier::fromString(vm, "type"_s), JSC::jsNontrivialString(vm, type));
    RETURN_IF_EXCEPTION(scope, JSC::JSValue());
    object->putDirect(vm, JSC::Identifier::fromString(vm, "value"_s), JSC::JSValue(value));
    RETURN_IF_EXCEPTION(scope, JSC::JSValue());
    return object;
}

template<>
JSC::JSValue jsValueForDecodedArgumentValue(JSC::JSGlobalObject* globalObject, bool value)
{
    auto& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    auto* object = JSC::constructEmptyObject(globalObject, globalObject->objectPrototype());
    RETURN_IF_EXCEPTION(scope, JSC::JSValue());
    object->putDirect(vm, JSC::Identifier::fromString(vm, "type"_s), JSC::jsNontrivialString(vm, "bool"_s));
    RETURN_IF_EXCEPTION(scope, JSC::JSValue());
    object->putDirect(vm, JSC::Identifier::fromString(vm, "value"_s), JSC::jsBoolean(value));
    RETURN_IF_EXCEPTION(scope, JSC::JSValue());
    return object;
}

template<>
JSC::JSValue jsValueForDecodedArgumentValue(JSC::JSGlobalObject* globalObject, double value)
{
    return jsValueForDecodedNumericArgumentValue(globalObject, value, "double"_s);
}

template<>
JSC::JSValue jsValueForDecodedArgumentValue(JSC::JSGlobalObject* globalObject, float value)
{
    return jsValueForDecodedNumericArgumentValue(globalObject, value, "float"_s);
}

template<>
JSC::JSValue jsValueForDecodedArgumentValue(JSC::JSGlobalObject* globalObject, int8_t value)
{
    return jsValueForDecodedNumericArgumentValue(globalObject, value, "int8_t"_s);
}

template<>
JSC::JSValue jsValueForDecodedArgumentValue(JSC::JSGlobalObject* globalObject, int16_t value)
{
    return jsValueForDecodedNumericArgumentValue(globalObject, value, "int16_t"_s);
}

template<>
JSC::JSValue jsValueForDecodedArgumentValue(JSC::JSGlobalObject* globalObject, int32_t value)
{
    return jsValueForDecodedNumericArgumentValue(globalObject, value, "int32_t"_s);
}

template<>
JSC::JSValue jsValueForDecodedArgumentValue(JSC::JSGlobalObject* globalObject, int64_t value)
{
    return jsValueForDecodedNumericArgumentValue(globalObject, value, "int64_t"_s);
}

template<>
JSC::JSValue jsValueForDecodedArgumentValue(JSC::JSGlobalObject* globalObject, uint8_t value)
{
    return jsValueForDecodedNumericArgumentValue(globalObject, value, "uint8_t"_s);
}

template<>
JSC::JSValue jsValueForDecodedArgumentValue(JSC::JSGlobalObject* globalObject, uint16_t value)
{
    return jsValueForDecodedNumericArgumentValue(globalObject, value, "uint16_t"_s);
}

template<>
JSC::JSValue jsValueForDecodedArgumentValue(JSC::JSGlobalObject* globalObject, uint32_t value)
{
    return jsValueForDecodedNumericArgumentValue(globalObject, value, "uint32_t"_s);
}

template<>
JSC::JSValue jsValueForDecodedArgumentValue(JSC::JSGlobalObject* globalObject, uint64_t value)
{
    return jsValueForDecodedNumericArgumentValue(globalObject, value, "uint64_t"_s);
}

template<>
JSC::JSValue jsValueForDecodedArgumentValue(JSC::JSGlobalObject* globalObject, size_t value)
{
    return jsValueForDecodedNumericArgumentValue(globalObject, value, "size_t"_s);
}

template<typename RectType>
JSC::JSValue jsValueForDecodedArgumentRect(JSC::JSGlobalObject* globalObject, const RectType& value, const String& type)
{
    auto& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    auto* object = JSC::constructEmptyObject(globalObject, globalObject->objectPrototype());
    RETURN_IF_EXCEPTION(scope, JSC::JSValue());
    object->putDirect(vm, JSC::Identifier::fromString(vm, "type"_s), JSC::jsNontrivialString(vm, type));
    RETURN_IF_EXCEPTION(scope, JSC::JSValue());
    object->putDirect(vm, JSC::Identifier::fromString(vm, "x"_s), JSC::JSValue(value.x()));
    RETURN_IF_EXCEPTION(scope, JSC::JSValue());
    object->putDirect(vm, JSC::Identifier::fromString(vm, "y"_s), JSC::JSValue(value.y()));
    RETURN_IF_EXCEPTION(scope, JSC::JSValue());
    object->putDirect(vm, JSC::Identifier::fromString(vm, "width"_s), JSC::JSValue(value.width()));
    RETURN_IF_EXCEPTION(scope, JSC::JSValue());
    object->putDirect(vm, JSC::Identifier::fromString(vm, "height"_s), JSC::JSValue(value.height()));
    RETURN_IF_EXCEPTION(scope, JSC::JSValue());
    return object;
}

template<>
JSC::JSValue jsValueForDecodedArgumentValue(JSC::JSGlobalObject* globalObject, WebCore::IntRect&& value)
{
    return jsValueForDecodedArgumentRect(globalObject, value, "IntRect"_s);
}

template<>
JSC::JSValue jsValueForDecodedArgumentValue(JSC::JSGlobalObject* globalObject, WebCore::FloatRect&& value)
{
    return jsValueForDecodedArgumentRect(globalObject, value, "FloatRect"_s);
}

template<>
JSC::JSValue jsValueForDecodedArgumentValue(JSC::JSGlobalObject* globalObject, WebCore::ExceptionData&& exceptionData)
{
    auto& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    auto* object = JSC::constructEmptyObject(globalObject, globalObject->objectPrototype());
    RETURN_IF_EXCEPTION(scope, JSC::JSValue());
    object->putDirect(vm, JSC::Identifier::fromString(vm, "type"_s), JSC::jsNontrivialString(vm, "ExceptionData"_s));
    RETURN_IF_EXCEPTION(scope, JSC::JSValue());
    auto& message = exceptionData.message;
    object->putDirect(vm, JSC::Identifier::fromString(vm, "message"_s), message.isNull() ? JSC::jsNull() : JSC::jsString(vm, message));
    RETURN_IF_EXCEPTION(scope, JSC::JSValue());
    object->putDirect(vm, JSC::Identifier::fromString(vm, "code"_s), JSC::jsNontrivialString(vm, WebCore::DOMException::description(exceptionData.code).name));
    RETURN_IF_EXCEPTION(scope, JSC::JSValue());
    return object;
}

}

#endif
