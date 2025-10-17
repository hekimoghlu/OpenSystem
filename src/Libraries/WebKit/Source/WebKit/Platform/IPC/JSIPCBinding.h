/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 19, 2024.
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
#pragma once

#if ENABLE(IPC_TESTING_API)

#include <wtf/Compiler.h>

#include "Decoder.h"
#include "HandleMessage.h"
#include <JavaScriptCore/JSArray.h>
#include <JavaScriptCore/JSArrayBuffer.h>
#include <JavaScriptCore/JSCJSValueInlines.h>
#include <JavaScriptCore/JSObject.h>
#include <JavaScriptCore/JSObjectInlines.h>
#include <JavaScriptCore/ObjectConstructor.h>
#include <WebCore/SharedMemory.h>
#include <wtf/ObjectIdentifier.h>
#include <wtf/text/WTFString.h>

namespace WTF {

class URL;

}

namespace WebCore {

class FloatRect;
class IntRect;
class RegistrableDomain;
struct ExceptionData;

}

namespace IPC {

class Semaphore;

template<typename T, std::enable_if_t<!std::is_arithmetic<T>::value && !std::is_enum<T>::value>* = nullptr>
JSC::JSValue jsValueForDecodedArgumentValue(JSC::JSGlobalObject*, T&&)
{
    // Report that we don't recognize this type.
    return JSC::JSValue();
}

template<> JSC::JSValue jsValueForDecodedArgumentValue(JSC::JSGlobalObject*, String&&);
template<> JSC::JSValue jsValueForDecodedArgumentValue(JSC::JSGlobalObject*, URL&&);
template<> JSC::JSValue jsValueForDecodedArgumentValue(JSC::JSGlobalObject*, WebCore::RegistrableDomain&&);

template<> JSC::JSValue jsValueForDecodedArgumentValue(JSC::JSGlobalObject*, IPC::Semaphore&&);
template<> JSC::JSValue jsValueForDecodedArgumentValue(JSC::JSGlobalObject*, WebCore::SharedMemory::Handle&&);

template<typename T, std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>
JSC::JSValue jsValueForDecodedArgumentValue(JSC::JSGlobalObject*, T)
{
}

template<typename E, std::enable_if_t<std::is_enum<E>::value>* = nullptr>
JSC::JSValue jsValueForDecodedArgumentValue(JSC::JSGlobalObject* globalObject, E value)
{
    return jsValueForDecodedArgumentValue(globalObject, static_cast<std::underlying_type_t<E>>(value));
}

template<> JSC::JSValue jsValueForDecodedArgumentValue(JSC::JSGlobalObject*, bool);
template<> JSC::JSValue jsValueForDecodedArgumentValue(JSC::JSGlobalObject*, double);
template<> JSC::JSValue jsValueForDecodedArgumentValue(JSC::JSGlobalObject*, float);
template<> JSC::JSValue jsValueForDecodedArgumentValue(JSC::JSGlobalObject*, int8_t);
template<> JSC::JSValue jsValueForDecodedArgumentValue(JSC::JSGlobalObject*, int16_t);
template<> JSC::JSValue jsValueForDecodedArgumentValue(JSC::JSGlobalObject*, int32_t);

template<> JSC::JSValue jsValueForDecodedArgumentValue(JSC::JSGlobalObject*, int64_t);
template<> JSC::JSValue jsValueForDecodedArgumentValue(JSC::JSGlobalObject*, uint8_t);
template<> JSC::JSValue jsValueForDecodedArgumentValue(JSC::JSGlobalObject*, uint16_t);
template<> JSC::JSValue jsValueForDecodedArgumentValue(JSC::JSGlobalObject*, uint32_t);
template<> JSC::JSValue jsValueForDecodedArgumentValue(JSC::JSGlobalObject*, uint64_t);
template<> JSC::JSValue jsValueForDecodedArgumentValue(JSC::JSGlobalObject*, size_t);

template<typename U>
JSC::JSValue jsValueForDecodedArgumentValue(JSC::JSGlobalObject* globalObject, ObjectIdentifier<U>&& value)
{
    return jsValueForDecodedArgumentValue(globalObject, value.toUInt64());
}

template<typename U>
JSC::JSValue jsValueForDecodedArgumentValue(JSC::JSGlobalObject* globalObject, AtomicObjectIdentifier<U>&& value)
{
    return jsValueForDecodedArgumentValue(globalObject, value.toUInt64());
}

template<> JSC::JSValue jsValueForDecodedArgumentValue(JSC::JSGlobalObject*, WebCore::IntRect&&);
template<> JSC::JSValue jsValueForDecodedArgumentValue(JSC::JSGlobalObject*, WebCore::FloatRect&&);
template<> JSC::JSValue jsValueForDecodedArgumentValue(JSC::JSGlobalObject*, WebCore::ExceptionData&&);

template<typename U>
JSC::JSValue jsValueForDecodedArgumentValue(JSC::JSGlobalObject* globalObject, OptionSet<U>&& value)
{    
    auto& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    auto result = jsValueForDecodedArgumentValue(globalObject, value.toRaw());
    RETURN_IF_EXCEPTION(scope, JSC::JSValue());
    result.getObject()->putDirect(vm, JSC::Identifier::fromString(vm, "isOptionSet"_s), JSC::jsBoolean(true));
    RETURN_IF_EXCEPTION(scope, JSC::JSValue());
    return result;
}

template<typename U>
JSC::JSValue jsValueForDecodedArgumentValue(JSC::JSGlobalObject* globalObject, std::optional<U>&& value)
{
    if (!value)
        return JSC::jsUndefined();
    return jsValueForDecodedArgumentValue(globalObject, std::forward<U>(*value));
}

template<typename... Elements>
std::optional<JSC::JSValue> putJSValueForDecodeArgumentInArray(JSC::JSGlobalObject*, IPC::Decoder&, JSC::JSArray*, size_t currentIndex, std::tuple<Elements...>*);

template<>
inline std::optional<JSC::JSValue> putJSValueForDecodeArgumentInArray(JSC::JSGlobalObject* globalObject, IPC::Decoder& decoder, JSC::JSArray* array, size_t currentIndex, std::tuple<>*)
{
    return JSC::JSValue { array };
}

template<typename T, typename... Elements>
std::optional<JSC::JSValue> putJSValueForDecodeArgumentInArray(JSC::JSGlobalObject* globalObject, IPC::Decoder& decoder, JSC::JSArray* array, size_t currentIndex, std::tuple<T, Elements...>*)
{
    auto startingBufferOffset = decoder.currentBufferOffset();
    std::optional<T> value;
    decoder >> value;
    if (!value)
        return std::nullopt;

    auto scope = DECLARE_THROW_SCOPE(globalObject->vm());
    auto jsValue = jsValueForDecodedArgumentValue(globalObject, WTFMove(*value));
    RETURN_IF_EXCEPTION(scope, std::nullopt);
    if (jsValue.isEmpty()) {
        // Create array buffers out of types we don't recognize.
        auto span = decoder.span().subspan(startingBufferOffset, decoder.currentBufferOffset() - startingBufferOffset);
        auto arrayBuffer = JSC::ArrayBuffer::create(span);
        if (auto* structure = globalObject->arrayBufferStructure(arrayBuffer->sharingMode()))
            jsValue = JSC::JSArrayBuffer::create(Ref { globalObject->vm() }, structure, WTFMove(arrayBuffer));
        RETURN_IF_EXCEPTION(scope, std::nullopt);
    }
    array->putDirectIndex(globalObject, currentIndex, jsValue);
    RETURN_IF_EXCEPTION(scope, std::nullopt);

    std::tuple<Elements...>* dummyArguments = nullptr;
    return putJSValueForDecodeArgumentInArray<Elements...>(globalObject, decoder, array, currentIndex + 1, dummyArguments);
}

template<typename T>
static std::optional<JSC::JSValue> jsValueForDecodedArguments(JSC::JSGlobalObject* globalObject, IPC::Decoder& decoder)
{
    auto scope = DECLARE_THROW_SCOPE(globalObject->vm());
    auto* array = JSC::constructEmptyArray(globalObject, nullptr);
    RETURN_IF_EXCEPTION(scope, JSC::JSValue());
    T* dummyArguments = nullptr;
    return putJSValueForDecodeArgumentInArray<>(globalObject, decoder, array, 0, dummyArguments);
}

// The bindings implementation will call the function templates below to decode a message.
// These function templates are specialized by each message in their own generated file.
// Each implementation will just call the above `jsValueForDecodedArguments()` function.
// This has the benefit that upon compilation, only the message receiver implementation files are
// recompiled when the message argument types change.
// The bindings implementation, e.g. the caller of jsValueForDecodedMessage<>, does not need
// to know all the message argument types, and need to be recompiled only when the message itself
// changes.
template<MessageName>
std::optional<JSC::JSValue> jsValueForDecodedMessage(JSC::JSGlobalObject*, IPC::Decoder&);
template<MessageName>
std::optional<JSC::JSValue> jsValueForDecodedMessageReply(JSC::JSGlobalObject*, IPC::Decoder&);

}

#endif
