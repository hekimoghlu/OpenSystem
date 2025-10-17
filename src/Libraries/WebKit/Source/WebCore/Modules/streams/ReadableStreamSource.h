/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 3, 2024.
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

#include "JSDOMPromiseDeferredForward.h"
#include "ReadableStreamDefaultController.h"
#include <wtf/AbstractRefCounted.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class ReadableStreamSource : public AbstractRefCounted {
public:
    WEBCORE_EXPORT ReadableStreamSource();
    WEBCORE_EXPORT virtual ~ReadableStreamSource();

    void start(ReadableStreamDefaultController&&, DOMPromiseDeferred<void>&&);
    void pull(DOMPromiseDeferred<void>&&);
    void cancel(JSC::JSValue);

    bool isPulling() const { return !!m_promise; }

protected:
    ReadableStreamDefaultController& controller() { return m_controller.value(); }
    const ReadableStreamDefaultController& controller() const { return m_controller.value(); }

    void startFinished();
    void pullFinished();
    void cancelFinished();
    WEBCORE_EXPORT void clean();

    virtual void setActive() = 0;
    virtual void setInactive() = 0;

    virtual void doStart() = 0;
    virtual void doPull() = 0;
    virtual void doCancel() = 0;

private:
    std::unique_ptr<DOMPromiseDeferred<void>> m_promise;
    std::optional<ReadableStreamDefaultController> m_controller;
};

class RefCountedReadableStreamSource
    : public ReadableStreamSource
    , public RefCounted<RefCountedReadableStreamSource> {
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }
};

class SimpleReadableStreamSource
    : public RefCountedReadableStreamSource
    , public CanMakeWeakPtr<SimpleReadableStreamSource> {
public:
    static Ref<SimpleReadableStreamSource> create() { return adoptRef(*new SimpleReadableStreamSource); }

    void close();
    void enqueue(JSC::JSValue);

private:
    SimpleReadableStreamSource() = default;

    // ReadableStreamSource
    void setActive() final { }
    void setInactive() final { }
    void doStart() final { }
    void doPull() final { }
    void doCancel() final;

    bool m_isCancelled { false };
};

} // namespace WebCore
