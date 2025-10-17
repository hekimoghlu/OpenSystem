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
#include "WritableStream.h"

#include "InternalWritableStream.h"
#include "JSDOMGlobalObject.h"
#include "JSWritableStream.h"
#include "JSWritableStreamSink.h"

namespace WebCore {

ExceptionOr<Ref<WritableStream>> WritableStream::create(JSC::JSGlobalObject& globalObject, std::optional<JSC::Strong<JSC::JSObject>>&& underlyingSink, std::optional<JSC::Strong<JSC::JSObject>>&& strategy)
{
    JSC::JSValue underlyingSinkValue = JSC::jsUndefined();
    if (underlyingSink)
        underlyingSinkValue = underlyingSink->get();

    JSC::JSValue strategyValue = JSC::jsUndefined();
    if (strategy)
        strategyValue = strategy->get();

    return create(globalObject, underlyingSinkValue, strategyValue);
}

WritableStream::~WritableStream() = default;

void WritableStream::lock()
{
    m_internalWritableStream->lock();
}

bool WritableStream::locked() const
{
    return m_internalWritableStream->locked();
}

InternalWritableStream& WritableStream::internalWritableStream()
{
    return m_internalWritableStream.get();
}

ExceptionOr<Ref<InternalWritableStream>> WritableStream::createInternalWritableStream(JSDOMGlobalObject& globalObject, Ref<WritableStreamSink>&& sink)
{
    return InternalWritableStream::createFromUnderlyingSink(globalObject, toJSNewlyCreated(&globalObject, &globalObject, WTFMove(sink)), JSC::jsUndefined());
}

ExceptionOr<Ref<WritableStream>> WritableStream::create(JSC::JSGlobalObject& globalObject, JSC::JSValue underlyingSink, JSC::JSValue strategy)
{
    auto result = InternalWritableStream::createFromUnderlyingSink(*JSC::jsCast<JSDOMGlobalObject*>(&globalObject), underlyingSink, strategy);
    if (result.hasException())
        return result.releaseException();

    return adoptRef(*new WritableStream(result.releaseReturnValue()));
}

ExceptionOr<Ref<WritableStream>> WritableStream::create(JSDOMGlobalObject& globalObject, Ref<WritableStreamSink>&& sink)
{
    return create(globalObject, toJSNewlyCreated(&globalObject, &globalObject, WTFMove(sink)), JSC::jsUndefined());
}

Ref<WritableStream> WritableStream::create(Ref<InternalWritableStream>&& internalWritableStream)
{
    return adoptRef(*new WritableStream(WTFMove(internalWritableStream)));
}

WritableStream::WritableStream(Ref<InternalWritableStream>&& internalWritableStream)
    : m_internalWritableStream(WTFMove(internalWritableStream))
{
}

void WritableStream::closeIfPossible()
{
    m_internalWritableStream->closeIfPossible();
}

JSC::JSValue JSWritableStream::abort(JSC::JSGlobalObject& globalObject, JSC::CallFrame& callFrame)
{
    return wrapped().internalWritableStream().abortForBindings(globalObject, callFrame.argument(0));
}

JSC::JSValue JSWritableStream::close(JSC::JSGlobalObject& globalObject, JSC::CallFrame&)
{
    return wrapped().internalWritableStream().closeForBindings(globalObject);
}

JSC::JSValue JSWritableStream::getWriter(JSC::JSGlobalObject& globalObject, JSC::CallFrame&)
{
    return wrapped().internalWritableStream().getWriter(globalObject);
}

} // namespace WebCore
