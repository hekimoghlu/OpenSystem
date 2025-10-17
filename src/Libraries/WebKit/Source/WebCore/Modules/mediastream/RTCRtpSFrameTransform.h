/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 18, 2025.
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
#include "EventTarget.h"
#include "JSDOMPromiseDeferredForward.h"
#include "RTCRtpSFrameTransformer.h"
#include <wtf/WeakPtr.h>

namespace JSC {
class JSGlobalObject;
};

namespace WebCore {

class CryptoKey;
class RTCRtpTransformBackend;
class ReadableStream;
class SimpleReadableStreamSource;
class WritableStream;

class RTCRtpSFrameTransform : public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<RTCRtpSFrameTransform>, public ActiveDOMObject, public EventTarget {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RTCRtpSFrameTransform);
public:
    enum class Role { Encrypt, Decrypt };
    using CompatibilityMode = RTCRtpSFrameTransformer::CompatibilityMode;
    struct Options {
        Role role { Role::Encrypt };
        uint64_t authenticationSize { 10 };
        CompatibilityMode compatibilityMode { CompatibilityMode::None };
    };

    static Ref<RTCRtpSFrameTransform> create(ScriptExecutionContext&, Options);
    ~RTCRtpSFrameTransform();

    void setEncryptionKey(CryptoKey&, std::optional<uint64_t>, DOMPromiseDeferred<void>&&);

    bool isAttached() const;
    void initializeBackendForReceiver(RTCRtpTransformBackend&);
    void initializeBackendForSender(RTCRtpTransformBackend&);
    void willClearBackend(RTCRtpTransformBackend&);

    WEBCORE_EXPORT void setCounterForTesting(uint64_t);
    WEBCORE_EXPORT uint64_t counterForTesting() const;
    WEBCORE_EXPORT uint64_t keyIdForTesting() const;

    ExceptionOr<RefPtr<ReadableStream>> readable();
    ExceptionOr<RefPtr<WritableStream>> writable();

    bool hasKey(uint64_t) const;

    // ActiveDOMObject.
    void ref() const final { ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr::ref(); }
    void deref() const final { ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr::deref(); }

private:
    RTCRtpSFrameTransform(ScriptExecutionContext&, Options);

    // ActiveDOMObject
    bool virtualHasPendingActivity() const final;

    // EventTarget
    enum EventTargetInterfaceType eventTargetInterface() const final { return EventTargetInterfaceType::RTCRtpSFrameTransform; }
    ScriptExecutionContext* scriptExecutionContext() const final { return ContextDestructionObserver::scriptExecutionContext(); }
    void refEventTarget() final { ref(); }
    void derefEventTarget() final { deref(); }

    enum class Side { Sender, Receiver };
    void initializeTransformer(RTCRtpTransformBackend&, Side);
    ExceptionOr<void> createStreams();

    bool m_isAttached { false };
    bool m_hasWritable { false };
    Ref<RTCRtpSFrameTransformer> m_transformer;
    RefPtr<ReadableStream> m_readable;
    RefPtr<WritableStream> m_writable;
    RefPtr<SimpleReadableStreamSource> m_readableStreamSource;
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
