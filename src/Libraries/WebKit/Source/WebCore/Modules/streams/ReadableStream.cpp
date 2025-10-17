/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 29, 2024.
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
#include "ReadableStream.h"

#include "JSReadableStream.h"
#include "JSReadableStreamSource.h"
#include "ScriptExecutionContext.h"

namespace WebCore {

ExceptionOr<Ref<ReadableStream>> ReadableStream::create(JSC::JSGlobalObject& globalObject, std::optional<JSC::Strong<JSC::JSObject>>&& underlyingSource, std::optional<JSC::Strong<JSC::JSObject>>&& strategy)
{
    JSC::JSValue underlyingSourceValue = JSC::jsUndefined();
    if (underlyingSource)
        underlyingSourceValue = underlyingSource->get();

    JSC::JSValue strategyValue = JSC::jsUndefined();
    if (strategy)
        strategyValue = strategy->get();

    return createFromJSValues(globalObject, underlyingSourceValue, strategyValue);
}

ExceptionOr<Ref<ReadableStream>> ReadableStream::createFromJSValues(JSC::JSGlobalObject& globalObject, JSC::JSValue underlyingSource, JSC::JSValue strategy)
{
    auto& jsDOMGlobalObject = *JSC::jsCast<JSDOMGlobalObject*>(&globalObject);
    RefPtr protectedContext { jsDOMGlobalObject.scriptExecutionContext() };
    auto result = InternalReadableStream::createFromUnderlyingSource(jsDOMGlobalObject, underlyingSource, strategy);
    if (result.hasException())
        return result.releaseException();

    return adoptRef(*new ReadableStream(result.releaseReturnValue()));
}

ExceptionOr<Ref<InternalReadableStream>> ReadableStream::createInternalReadableStream(JSDOMGlobalObject& globalObject, Ref<ReadableStreamSource>&& source)
{
    return InternalReadableStream::createFromUnderlyingSource(globalObject, toJSNewlyCreated(&globalObject, &globalObject, WTFMove(source)), JSC::jsUndefined());
}

ExceptionOr<Ref<ReadableStream>> ReadableStream::create(JSDOMGlobalObject& globalObject, Ref<ReadableStreamSource>&& source)
{
    return createFromJSValues(globalObject, toJSNewlyCreated(&globalObject, &globalObject, WTFMove(source)), JSC::jsUndefined());
}

Ref<ReadableStream> ReadableStream::create(Ref<InternalReadableStream>&& internalReadableStream)
{
    return adoptRef(*new ReadableStream(WTFMove(internalReadableStream)));
}

ReadableStream::ReadableStream(Ref<InternalReadableStream>&& internalReadableStream)
    : m_internalReadableStream(WTFMove(internalReadableStream))
{
}

ExceptionOr<Vector<Ref<ReadableStream>>> ReadableStream::tee(bool shouldClone)
{
    auto result = m_internalReadableStream->tee(shouldClone);
    if (result.hasException())
        return result.releaseException();

    auto pair = result.releaseReturnValue();

    return Vector {
        ReadableStream::create(WTFMove(pair.first)),
        ReadableStream::create(WTFMove(pair.second))
    };
}

JSC::JSValue JSReadableStream::cancel(JSC::JSGlobalObject& globalObject, JSC::CallFrame& callFrame)
{
    return wrapped().internalReadableStream().cancelForBindings(globalObject, callFrame.argument(0));
}

JSC::JSValue JSReadableStream::getReader(JSC::JSGlobalObject& globalObject, JSC::CallFrame& callFrame)
{
    return wrapped().internalReadableStream().getReader(globalObject, callFrame.argument(0));
}

JSC::JSValue JSReadableStream::pipeTo(JSC::JSGlobalObject& globalObject, JSC::CallFrame& callFrame)
{
    return wrapped().internalReadableStream().pipeTo(globalObject, callFrame.argument(0), callFrame.argument(1));
}

JSC::JSValue JSReadableStream::pipeThrough(JSC::JSGlobalObject& globalObject, JSC::CallFrame& callFrame)
{
    return wrapped().internalReadableStream().pipeThrough(globalObject, callFrame.argument(0), callFrame.argument(1));
}

} // namespace WebCore
