/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 20, 2025.
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
#include "StructuredClone.h"

#include "JSDOMBinding.h"
#include "JSDOMConvertBufferSource.h"
#include "JSDOMExceptionHandling.h"
#include <JavaScriptCore/JSTypedArrays.h>

namespace WebCore {
using namespace JSC;

enum class CloneMode {
    Full,
    Partial,
};

static EncodedJSValue cloneArrayBufferImpl(JSGlobalObject* lexicalGlobalObject, CallFrame* callFrame, CloneMode mode)
{
    VM& vm = lexicalGlobalObject->vm();

    ASSERT(lexicalGlobalObject);
    ASSERT(callFrame->argumentCount());
    ASSERT(callFrame->lexicalGlobalObject(vm) == lexicalGlobalObject);

    auto* buffer = toUnsharedArrayBuffer(vm, callFrame->uncheckedArgument(0));
    if (!buffer) {
        auto scope = DECLARE_THROW_SCOPE(vm);
        throwDataCloneError(*lexicalGlobalObject, scope);
        return { };
    }
    if (mode == CloneMode::Partial) {
        ASSERT(callFrame->argumentCount() == 3);
        int srcByteOffset = static_cast<int>(callFrame->uncheckedArgument(1).toNumber(lexicalGlobalObject));
        int srcLength = static_cast<int>(callFrame->uncheckedArgument(2).toNumber(lexicalGlobalObject));
        return JSValue::encode(JSArrayBuffer::create(lexicalGlobalObject->vm(), lexicalGlobalObject->arrayBufferStructure(ArrayBufferSharingMode::Default), buffer->slice(srcByteOffset, srcByteOffset + srcLength)));
    }
    return JSValue::encode(JSArrayBuffer::create(lexicalGlobalObject->vm(), lexicalGlobalObject->arrayBufferStructure(ArrayBufferSharingMode::Default), ArrayBuffer::tryCreate(buffer->span())));
}

JSC_DEFINE_HOST_FUNCTION(cloneArrayBuffer, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    return cloneArrayBufferImpl(globalObject, callFrame, CloneMode::Partial);
}

JSC_DEFINE_HOST_FUNCTION(structuredCloneForStream, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    ASSERT(callFrame);
    ASSERT(callFrame->argumentCount());

    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue value = callFrame->uncheckedArgument(0);

    auto cloneArrayBuffer = [&](ArrayBuffer& buffer) -> RefPtr<ArrayBuffer> {
        auto scope = DECLARE_THROW_SCOPE(vm);
        size_t byteLength = buffer.byteLength();
        auto result = ArrayBuffer::tryCreate(byteLength, 1, buffer.maxByteLength());
        if (UNLIKELY(!result)) {
            throwOutOfMemoryError(globalObject, scope);
            return nullptr;
        }

        memcpySpan(result->mutableSpan(), buffer.span().first(byteLength));
        return result;
    };

    if (value.inherits<JSArrayBuffer>()) {
        auto* buffer = toUnsharedArrayBuffer(vm, value);
        if (UNLIKELY(!buffer || buffer->isDetached())) {
            throwDataCloneError(*globalObject, scope);
            return { };
        }

        auto result = cloneArrayBuffer(*buffer);
        RETURN_IF_EXCEPTION(scope, { });

        return JSValue::encode(JSArrayBuffer::create(globalObject->vm(), globalObject->arrayBufferStructure(ArrayBufferSharingMode::Default), result.releaseNonNull()));
    }

    if (value.inherits<JSArrayBufferView>()) {
        auto* bufferView = jsCast<JSArrayBufferView*>(value);
        ASSERT(bufferView);

        auto* buffer = bufferView->unsharedBuffer();
        if (UNLIKELY(!buffer || buffer->isDetached())) {
            throwDataCloneError(*globalObject, scope);
            return { };
        }

        auto bufferClone = cloneArrayBuffer(*buffer);
        RETURN_IF_EXCEPTION(scope, { });

        size_t byteOffset = 0;
        std::optional<size_t> length;
        if (bufferView->isResizableOrGrowableShared()) {
            byteOffset = bufferView->byteOffsetRaw();
            if (!bufferView->isAutoLength())
                length = bufferView->lengthRaw();
        } else {
            byteOffset = bufferView->byteOffset();
            length = bufferView->length();
        }

        switch (typedArrayType(bufferView->type())) {
#define CLONE_TYPED_ARRAY(name) \
        case Type##name: { \
            RELEASE_AND_RETURN(scope, JSValue::encode(toJS(globalObject, globalObject, name##Array::wrappedAs(bufferClone.releaseNonNull(), byteOffset, length).get()))); \
        }
        FOR_EACH_TYPED_ARRAY_TYPE_EXCLUDING_DATA_VIEW(CLONE_TYPED_ARRAY)
#undef CLONE_TYPED_ARRAY
        case TypeDataView: {
            RELEASE_AND_RETURN(scope, JSValue::encode(toJS(globalObject, globalObject, DataView::wrappedAs(bufferClone.releaseNonNull(), byteOffset, length).get())));
        }
        default:
            RELEASE_ASSERT_NOT_REACHED();
            break;
        }
    }

    throwTypeError(globalObject, scope, "structuredClone not implemented for non-ArrayBuffer / non-ArrayBufferView"_s);
    return { };
}

} // namespace WebCore
