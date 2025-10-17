/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 7, 2025.
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
#include "ReadableStreamSink.h"

#include "DOMException.h"
#include "JSDOMGlobalObject.h"
#include "ReadableStream.h"
#include "SharedBuffer.h"
#include <JavaScriptCore/Uint8Array.h>

namespace WebCore {

ReadableStreamToSharedBufferSink::ReadableStreamToSharedBufferSink(Callback&& callback)
    : m_callback { WTFMove(callback) }
{
}

void ReadableStreamToSharedBufferSink::pipeFrom(ReadableStream& stream)
{
    stream.pipeTo(*this);
}

void ReadableStreamToSharedBufferSink::enqueue(const Ref<JSC::Uint8Array>& buffer)
{
    if (!buffer->byteLength())
        return;

    if (m_callback) {
        auto chunk = buffer->span();
        m_callback(&chunk);
    }
}

void ReadableStreamToSharedBufferSink::close()
{
    if (!m_callback)
        return;

    auto callback = std::exchange(m_callback, { });
    callback(nullptr);
}

void ReadableStreamToSharedBufferSink::error(String&& message)
{
    if (!m_callback)
        return;

    auto callback = std::exchange(m_callback, { });
    callback(Exception { ExceptionCode::TypeError, WTFMove(message) });
}

} // namespace WebCore
