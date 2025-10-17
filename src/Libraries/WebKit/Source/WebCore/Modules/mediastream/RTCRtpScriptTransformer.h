/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 13, 2023.
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

#if ENABLE(WEB_RTC)

#include "ActiveDOMObject.h"
#include "ExceptionOr.h"
#include "JSDOMPromiseDeferredForward.h"
#include "RTCRtpTransformBackend.h"
#include <JavaScriptCore/JSCJSValue.h>
#include <wtf/Deque.h>
#include <wtf/ObjectIdentifier.h>
#include <wtf/RefCounted.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class FrameRateMonitor;
class MessagePort;
class ReadableStream;
class ScriptExecutionContext;
class RTCRtpTransformBackend;
class SerializedScriptValue;
class SimpleReadableStreamSource;
class WritableStream;

struct MessageWithMessagePorts;

enum class RTCRtpScriptTransformerIdentifierType { };
using RTCRtpScriptTransformerIdentifier = AtomicObjectIdentifier<RTCRtpScriptTransformerIdentifierType>;

class RTCRtpScriptTransformer
    : public RefCounted<RTCRtpScriptTransformer>
    , public ActiveDOMObject
    , public CanMakeWeakPtr<RTCRtpScriptTransformer> {
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    static ExceptionOr<Ref<RTCRtpScriptTransformer>> create(ScriptExecutionContext&, MessageWithMessagePorts&&);
    ~RTCRtpScriptTransformer();

    ReadableStream& readable();
    ExceptionOr<Ref<WritableStream>> writable();
    JSC::JSValue options(JSC::JSGlobalObject&);

    void generateKeyFrame(Ref<DeferredPromise>&&);
    void sendKeyFrameRequest(Ref<DeferredPromise>&&);

    void startPendingActivity() { m_pendingActivity = makePendingActivity(*this); }
    void start(Ref<RTCRtpTransformBackend>&&);

    enum class ClearCallback : bool { No, Yes };
    void clear(ClearCallback);

private:
    RTCRtpScriptTransformer(ScriptExecutionContext&, Ref<SerializedScriptValue>&&, Vector<Ref<MessagePort>>&&, Ref<ReadableStream>&&, Ref<SimpleReadableStreamSource>&&);

    // ActiveDOMObject.
    void stop() final { stopPendingActivity(); }

    void stopPendingActivity() { auto pendingActivity = WTFMove(m_pendingActivity); }

    void enqueueFrame(ScriptExecutionContext&, Ref<RTCRtpTransformableFrame>&&);

    Ref<SerializedScriptValue> m_options;
    Vector<Ref<MessagePort>> m_ports;

    Ref<SimpleReadableStreamSource> m_readableSource;
    Ref<ReadableStream> m_readable;
    RefPtr<WritableStream> m_writable;

    RefPtr<RTCRtpTransformBackend> m_backend;
    RefPtr<PendingActivity<RTCRtpScriptTransformer>> m_pendingActivity;

    Deque<Ref<DeferredPromise>> m_pendingKeyFramePromises;

#if !RELEASE_LOG_DISABLED
    bool m_enableAdditionalLogging { false };
    RTCRtpScriptTransformerIdentifier m_identifier;
    std::unique_ptr<FrameRateMonitor> m_readableFrameRateMonitor;
    std::unique_ptr<FrameRateMonitor> m_writableFrameRateMonitor;
#endif
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
