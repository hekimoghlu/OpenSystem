/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 3, 2024.
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
#include "ReadableStreamSource.h"

#include "JSDOMPromiseDeferred.h"

namespace WebCore {

ReadableStreamSource::ReadableStreamSource() = default;
ReadableStreamSource::~ReadableStreamSource() = default;

void ReadableStreamSource::start(ReadableStreamDefaultController&& controller, DOMPromiseDeferred<void>&& promise)
{
    ASSERT(!m_promise);
    m_promise = makeUnique<DOMPromiseDeferred<void>>(WTFMove(promise));
    m_controller = WTFMove(controller);

    setActive();
    doStart();
}

void ReadableStreamSource::pull(DOMPromiseDeferred<void>&& promise)
{
    ASSERT(!m_promise);
    ASSERT(m_controller);

    m_promise = makeUnique<DOMPromiseDeferred<void>>(WTFMove(promise));

    setActive();
    doPull();
}

void ReadableStreamSource::startFinished()
{
    ASSERT(m_promise);
    m_promise->resolve();
    m_promise = nullptr;
    setInactive();
}

void ReadableStreamSource::pullFinished()
{
    ASSERT(m_promise);
    m_promise->resolve();
    m_promise = nullptr;
    setInactive();
}

void ReadableStreamSource::cancel(JSC::JSValue)
{
    clean();
    doCancel();
}

void ReadableStreamSource::clean()
{
    if (m_promise) {
        m_promise = nullptr;
        setInactive();
    }
}

void SimpleReadableStreamSource::doCancel()
{
    m_isCancelled = true;
}

void SimpleReadableStreamSource::close()
{
    if (!m_isCancelled)
        controller().close();
}

void SimpleReadableStreamSource::enqueue(JSC::JSValue value)
{
    if (!m_isCancelled)
        controller().enqueue(value);
}

} // namespace WebCore
