/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 9, 2025.
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

#include "ExceptionOr.h"
#include "JSDOMPromiseDeferred.h"
#include "ScriptExecutionContext.h"
#include <wtf/RefCounted.h>
#include <wtf/text/WTFString.h>

namespace JSC {
class JSValue;
}

namespace WebCore {

class WritableStreamSink : public RefCounted<WritableStreamSink> {
public:
    virtual ~WritableStreamSink() = default;

    virtual void write(ScriptExecutionContext&, JSC::JSValue, DOMPromiseDeferred<void>&&) = 0;
    virtual void close() = 0;
    virtual void error(String&&) = 0;
};

class SimpleWritableStreamSink : public WritableStreamSink {
public:
    using WriteCallback = Function<ExceptionOr<void>(ScriptExecutionContext&, JSC::JSValue)>;
    static Ref<SimpleWritableStreamSink> create(WriteCallback&& writeCallback) { return adoptRef(*new SimpleWritableStreamSink(WTFMove(writeCallback))); }

private:
    explicit SimpleWritableStreamSink(WriteCallback&&);

    void write(ScriptExecutionContext&, JSC::JSValue, DOMPromiseDeferred<void>&&) final;
    void close() final { }
    void error(String&&) final { }

    WriteCallback m_writeCallback;
};

inline SimpleWritableStreamSink::SimpleWritableStreamSink(WriteCallback&& writeCallback)
    : m_writeCallback(WTFMove(writeCallback))
{
}

inline void SimpleWritableStreamSink::write(ScriptExecutionContext& context, JSC::JSValue value, DOMPromiseDeferred<void>&& promise)
{
    promise.settle(m_writeCallback(context, value));
}

} // namespace WebCore
