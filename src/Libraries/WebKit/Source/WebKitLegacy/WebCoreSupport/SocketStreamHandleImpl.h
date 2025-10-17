/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 9, 2024.
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

#include "SocketStreamHandle.h"
#include <pal/SessionID.h>
#include <wtf/RetainPtr.h>
#include <wtf/StreamBuffer.h>

typedef struct __CFHTTPMessage* CFHTTPMessageRef;

namespace WebCore {

class Credential;
class StorageSessionProvider;
class ProtectionSpace;
class SocketStreamHandleClient;

class SocketStreamHandleImpl : public SocketStreamHandle {
public:
    static Ref<SocketStreamHandleImpl> create(const URL& url, SocketStreamHandleClient& client, PAL::SessionID sessionID, const String& credentialPartition, SourceApplicationAuditToken&& auditData, const StorageSessionProvider* provider, bool shouldAcceptInsecureCertificates) { return adoptRef(*new SocketStreamHandleImpl(url, client, sessionID, credentialPartition, WTFMove(auditData), provider, shouldAcceptInsecureCertificates)); }

    virtual ~SocketStreamHandleImpl();

    static void setLegacyTLSEnabled(bool);

    void platformSend(std::span<const uint8_t> data, Function<void(bool)>&&) final;
    void platformSendHandshake(std::span<const uint8_t> data, const std::optional<CookieRequestHeaderFieldProxy>&, Function<void(bool, bool)>&&) final;
    void platformClose() final;
private:
    size_t bufferedAmount() final;
    std::optional<size_t> platformSendInternal(std::span<const uint8_t>);
    bool sendPendingData();

    SocketStreamHandleImpl(const URL&, SocketStreamHandleClient&, PAL::SessionID, const String& credentialPartition, SourceApplicationAuditToken&&, const StorageSessionProvider*, bool shouldAcceptInsecureCertificates);
    void createStreams();
    void scheduleStreams();
    void chooseProxy();
    void chooseProxyFromArray(CFArrayRef);
    void executePACFileURL(CFURLRef);
    void removePACRunLoopSource();
    RetainPtr<CFRunLoopSourceRef> m_pacRunLoopSource;
    static void pacExecutionCallback(void* client, CFArrayRef proxyList, CFErrorRef);
    static CFStringRef copyPACExecutionDescription(void*);

    bool shouldUseSSL() const { return m_url.protocolIs("wss"_s); }
    unsigned short port() const;

    void addCONNECTCredentials(CFHTTPMessageRef response);

    static void* retainSocketStreamHandle(void*);
    static void releaseSocketStreamHandle(void*);
    static CFStringRef copyCFStreamDescription(void*);
    static void readStreamCallback(CFReadStreamRef, CFStreamEventType, void*);
    static void writeStreamCallback(CFWriteStreamRef, CFStreamEventType, void*);
    void readStreamCallback(CFStreamEventType);
    void writeStreamCallback(CFStreamEventType);

    void reportErrorToClient(CFErrorRef);

    bool getStoredCONNECTProxyCredentials(const ProtectionSpace&, String& login, String& password);

    enum ConnectingSubstate { New, ExecutingPACFile, WaitingForCredentials, WaitingForConnect, Connected };
    ConnectingSubstate m_connectingSubstate;

    enum ConnectionType { Unknown, Direct, SOCKSProxy, CONNECTProxy };
    ConnectionType m_connectionType;
    RetainPtr<CFStringRef> m_proxyHost;
    RetainPtr<CFNumberRef> m_proxyPort;

    RetainPtr<CFHTTPMessageRef> m_proxyResponseMessage;
    bool m_sentStoredCredentials;
    bool m_shouldAcceptInsecureCertificates;
    RetainPtr<CFReadStreamRef> m_readStream;
    RetainPtr<CFWriteStreamRef> m_writeStream;

    RetainPtr<CFURLRef> m_httpsURL; // ws(s): replaced with https:
    String m_credentialPartition;
    SourceApplicationAuditToken m_auditData;
    RefPtr<const StorageSessionProvider> m_storageSessionProvider;

    StreamBuffer<uint8_t, 1024 * 1024> m_buffer;
    static const unsigned maxBufferSize = 100 * 1024 * 1024;
};

} // namespace WebCore
