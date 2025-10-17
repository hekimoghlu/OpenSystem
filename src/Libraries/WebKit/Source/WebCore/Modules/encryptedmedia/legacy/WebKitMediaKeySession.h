/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 17, 2023.
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

#if ENABLE(LEGACY_ENCRYPTED_MEDIA)

#include "ActiveDOMObject.h"
#include "ContextDestructionObserverInlines.h"
#include "EventTarget.h"
#include "ExceptionOr.h"
#include "LegacyCDMSession.h"
#include "Timer.h"
#include <JavaScriptCore/Forward.h>
#include <wtf/Deque.h>

namespace WebCore {

class WebKitMediaKeyError;
class WebKitMediaKeys;

class WebKitMediaKeySession final : public RefCounted<WebKitMediaKeySession>, public EventTarget, public ActiveDOMObject, private LegacyCDMSessionClient {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(WebKitMediaKeySession);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    static Ref<WebKitMediaKeySession> create(Document&, WebKitMediaKeys&, const String& keySystem);
    ~WebKitMediaKeySession();

    WebKitMediaKeyError* error() { return m_error.get(); }
    const String& keySystem() const { return m_keySystem; }
    const String& sessionId() const { return m_sessionId; }
    ExceptionOr<void> update(Ref<Uint8Array>&& key);
    void close();

    LegacyCDMSession* session() { return m_session.get(); }

    void detachKeys() { m_keys = nullptr; }

    void generateKeyRequest(const String& mimeType, Ref<Uint8Array>&& initData);
    RefPtr<ArrayBuffer> cachedKeyForKeyId(const String& keyId) const;

private:
    WebKitMediaKeySession(Document&, WebKitMediaKeys&, const String& keySystem);
    void keyRequestTimerFired();
    void addKeyTimerFired();

    void sendMessage(Uint8Array*, String destinationURL) final;
    void sendError(MediaKeyErrorCode, uint32_t systemCode) final;
    String mediaKeysStorageDirectory() const final;

    void refEventTarget() final { ref(); }
    void derefEventTarget() final { deref(); }

    // ActiveDOMObject.
    void stop() final;
    bool virtualHasPendingActivity() const final;

    enum EventTargetInterfaceType eventTargetInterface() const final { return EventTargetInterfaceType::WebKitMediaKeySession; }
    ScriptExecutionContext* scriptExecutionContext() const final { return ActiveDOMObject::scriptExecutionContext(); }

#if !RELEASE_LOG_DISABLED
    const Logger& logger() const final { return m_logger; }
    uint64_t logIdentifier() const final { return m_logIdentifier; }
    ASCIILiteral logClassName() const { return "WebKitMediaKeySession"_s; }
    WTFLogChannel& logChannel() const;

    Ref<Logger> m_logger;
    const uint64_t m_logIdentifier;
#endif

    WebKitMediaKeys* m_keys;
    String m_keySystem;
    String m_sessionId;
    RefPtr<WebKitMediaKeyError> m_error;
    RefPtr<LegacyCDMSession> m_session;

    struct PendingKeyRequest {
        String mimeType;
        Ref<Uint8Array> initData;
    };
    Deque<PendingKeyRequest> m_pendingKeyRequests;
    Timer m_keyRequestTimer;

    Deque<Ref<Uint8Array>> m_pendingKeys;
    Timer m_addKeyTimer;
};

}

#endif // ENABLE(LEGACY_ENCRYPTED_MEDIA)
