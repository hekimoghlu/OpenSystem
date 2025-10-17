/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 13, 2023.
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
#include "RTCRtpScriptTransformer.h"
#include <JavaScriptCore/JSCJSValue.h>
#include <JavaScriptCore/JSObject.h>
#include <JavaScriptCore/Strong.h>
#include <wtf/Lock.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeRefCounted.h>

namespace WebCore {

class RTCRtpScriptTransformer;
class RTCRtpTransformBackend;
class Worker;

class RTCRtpScriptTransform final
    : public ThreadSafeRefCounted<RTCRtpScriptTransform, WTF::DestructionThread::Main>
    , public ActiveDOMObject {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RTCRtpScriptTransform);
public:
    static ExceptionOr<Ref<RTCRtpScriptTransform>> create(JSC::JSGlobalObject&, Worker&, JSC::JSValue, Vector<JSC::Strong<JSC::JSObject>>&&);
    ~RTCRtpScriptTransform();

    // ActiveDOMObject.
    void ref() const final { ThreadSafeRefCounted::ref(); }
    void deref() const final { ThreadSafeRefCounted::deref(); }

    void setTransformer(RTCRtpScriptTransformer&);

    bool isAttached() const { return m_isAttached; }
    void initializeBackendForReceiver(RTCRtpTransformBackend&);
    void initializeBackendForSender(RTCRtpTransformBackend&);
    void willClearBackend(RTCRtpTransformBackend&);
    void backendTransferedToNewTransform() { clear(RTCRtpScriptTransformer::ClearCallback::No); }

private:
    RTCRtpScriptTransform(ScriptExecutionContext&, Ref<Worker>&&);

    void initializeTransformer(RTCRtpTransformBackend&);
    bool setupTransformer(Ref<RTCRtpTransformBackend>&&);
    void clear(RTCRtpScriptTransformer::ClearCallback);

    Ref<Worker> m_worker;

    bool m_isAttached { false };
    RefPtr<RTCRtpTransformBackend> m_backend;

    Lock m_transformerLock;
    bool m_isTransformerInitialized WTF_GUARDED_BY_LOCK(m_transformerLock) { false };
    WeakPtr<RTCRtpScriptTransformer> m_transformer WTF_GUARDED_BY_LOCK(m_transformerLock);
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
