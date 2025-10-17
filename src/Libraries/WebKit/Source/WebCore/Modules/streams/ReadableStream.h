/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 28, 2024.
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

#include "InternalReadableStream.h"
#include <JavaScriptCore/Strong.h>
#include <wtf/RefCounted.h>

namespace WebCore {

class InternalReadableStream;
class JSDOMGlobalObject;
class ReadableStreamSource;

class ReadableStream : public RefCounted<ReadableStream> {
public:
    static ExceptionOr<Ref<ReadableStream>> create(JSC::JSGlobalObject&, std::optional<JSC::Strong<JSC::JSObject>>&&, std::optional<JSC::Strong<JSC::JSObject>>&&);
    static ExceptionOr<Ref<ReadableStream>> create(JSDOMGlobalObject&, Ref<ReadableStreamSource>&&);
    static Ref<ReadableStream> create(Ref<InternalReadableStream>&&);

    ~ReadableStream() = default;

    void lock() { m_internalReadableStream->lock(); }
    bool isLocked() const { return m_internalReadableStream->isLocked(); }
    bool isDisturbed() const { return m_internalReadableStream->isDisturbed(); }
    void cancel(Exception&& exception) { m_internalReadableStream->cancel(WTFMove(exception)); }
    void pipeTo(ReadableStreamSink& sink) { m_internalReadableStream->pipeTo(sink); }
    ExceptionOr<Vector<Ref<ReadableStream>>> tee(bool shouldClone = false);

    InternalReadableStream& internalReadableStream() { return m_internalReadableStream.get(); }

protected:
    static ExceptionOr<Ref<ReadableStream>> createFromJSValues(JSC::JSGlobalObject&, JSC::JSValue, JSC::JSValue);
    static ExceptionOr<Ref<InternalReadableStream>> createInternalReadableStream(JSDOMGlobalObject&, Ref<ReadableStreamSource>&&);
    explicit ReadableStream(Ref<InternalReadableStream>&&);

private:
    Ref<InternalReadableStream> m_internalReadableStream;
};

} // namespace WebCore
