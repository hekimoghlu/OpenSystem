/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 18, 2025.
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
#include <JavaScriptCore/Forward.h>
#include <span>
#include <wtf/Function.h>
#include <wtf/RefCounted.h>

namespace WebCore {

class BufferSource;
class ReadableStream;

class ReadableStreamSink : public RefCounted<ReadableStreamSink> {
public:
    virtual ~ReadableStreamSink() = default;

    virtual void enqueue(const Ref<JSC::Uint8Array>&) = 0;
    virtual void close() = 0;
    virtual void error(String&&) = 0;
};

class ReadableStreamToSharedBufferSink final : public ReadableStreamSink {
public:
    using Callback = Function<void(ExceptionOr<std::span<const uint8_t>*>&&)>;
    static Ref<ReadableStreamToSharedBufferSink> create(Callback&& callback) { return adoptRef(*new ReadableStreamToSharedBufferSink(WTFMove(callback))); }
    void pipeFrom(ReadableStream&);
    void clearCallback() { m_callback = { }; }

private:
    explicit ReadableStreamToSharedBufferSink(Callback&&);

    void enqueue(const Ref<JSC::Uint8Array>&) final;
    void close() final;
    void error(String&&) final;

    Callback m_callback;
};

} // namespace WebCore
