/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 12, 2023.
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

#include "NetworkDataTask.h"
#include "NetworkLoadParameters.h"
#include <WebCore/FrameIdentifier.h>
#include <WebCore/NetworkLoadMetrics.h>
#include <WebCore/PageIdentifier.h>
#include <WebCore/ProtectionSpace.h>
#include <WebCore/ResourceResponse.h>
#include <wtf/RunLoop.h>
#include <wtf/glib/GRefPtr.h>
#include <wtf/text/CString.h>

namespace WebKit {

class NetworkDataTaskSoup final : public NetworkDataTask {
public:
    static Ref<NetworkDataTask> create(NetworkSession& session, NetworkDataTaskClient& client, const NetworkLoadParameters& parameters)
    {
        return adoptRef(*new NetworkDataTaskSoup(session, client, parameters));
    }

    ~NetworkDataTaskSoup();

private:
    NetworkDataTaskSoup(NetworkSession&, NetworkDataTaskClient&, const NetworkLoadParameters&);

    void cancel() override;
    void resume() override;
    void invalidateAndCancel() override;
    NetworkDataTask::State state() const override;
    void setTimingAllowFailedFlag() final;

    void setPendingDownloadLocation(const String&, SandboxExtension::Handle&&, bool /*allowOverwrite*/) override;
    String suggestedFilename() const override;

    void setPriority(WebCore::ResourceLoadPriority) override;

    void timeoutFired();
    void startTimeout();
    void stopTimeout();

    enum class WasBlockingCookies : bool { No, Yes };
    void createRequest(WebCore::ResourceRequest&&, WasBlockingCookies);
    void clearRequest();

    struct SendRequestData {
        WTF_MAKE_STRUCT_FAST_ALLOCATED;
        GRefPtr<SoupMessage> soupMessage;
        RefPtr<NetworkDataTaskSoup> task;
    };
    static void sendRequestCallback(SoupSession*, GAsyncResult*, SendRequestData*);
    void didSendRequest(GRefPtr<GInputStream>&&);
    void dispatchDidReceiveResponse();
    void dispatchDidCompleteWithError(const WebCore::ResourceError&);

#if !USE(SOUP2)
    static void preconnectCallback(SoupSession*, GAsyncResult*, NetworkDataTaskSoup*);
#endif

#if USE(SOUP2)
    static gboolean tlsConnectionAcceptCertificateCallback(GTlsConnection*, GTlsCertificate*, GTlsCertificateFlags, NetworkDataTaskSoup*);
#else
    static gboolean acceptCertificateCallback(SoupMessage*, GTlsCertificate*, GTlsCertificateFlags, NetworkDataTaskSoup*);
#endif
    bool acceptCertificate(GTlsCertificate*, GTlsCertificateFlags);

    static void didSniffContentCallback(SoupMessage*, const char* contentType, GHashTable* parameters, NetworkDataTaskSoup*);
    void didSniffContent(CString&&);

    bool persistentCredentialStorageEnabled() const;
    void applyAuthenticationToRequest(WebCore::ResourceRequest&);
#if USE(SOUP2)
    static void authenticateCallback(SoupSession*, SoupMessage*, SoupAuth*, gboolean retrying, NetworkDataTaskSoup*);
#else
    static gboolean authenticateCallback(SoupMessage*, SoupAuth*, gboolean retrying, NetworkDataTaskSoup*);
#endif
    void authenticate(WebCore::AuthenticationChallenge&&);
    void continueAuthenticate(WebCore::AuthenticationChallenge&&);
    void completeAuthentication(const WebCore::AuthenticationChallenge&, const WebCore::Credential&);
    void cancelAuthentication(const WebCore::AuthenticationChallenge&);

    static void skipInputStreamForRedirectionCallback(GInputStream*, GAsyncResult*, NetworkDataTaskSoup*);
    void skipInputStreamForRedirection();
    void didFinishSkipInputStreamForRedirection();
    bool shouldStartHTTPRedirection();
    void continueHTTPRedirection();

    static void readCallback(GInputStream*, GAsyncResult*, NetworkDataTaskSoup*);
    void read();
    void didRead(gssize bytesRead);
    void didFinishRead();

    static void requestNextPartCallback(SoupMultipartInputStream*, GAsyncResult*, NetworkDataTaskSoup*);
    void requestNextPart();
    void didRequestNextPart(GRefPtr<GInputStream>&&);
    void didFinishRequestNextPart();

    static void gotHeadersCallback(SoupMessage*, NetworkDataTaskSoup*);
    void didGetHeaders();

#if USE(SOUP2)
    static void wroteBodyDataCallback(SoupMessage*, SoupBuffer*, NetworkDataTaskSoup*);
#else
    static void wroteBodyDataCallback(SoupMessage*, unsigned, NetworkDataTaskSoup*);
#endif
    void didWriteBodyData(uint64_t bytesSent);

#if !USE(SOUP2)
    static void wroteHeadersCallback(SoupMessage*, NetworkDataTaskSoup*);
    static void wroteBodyCallback(SoupMessage*, NetworkDataTaskSoup*);
    static void gotBodyCallback(SoupMessage*, NetworkDataTaskSoup*);
    static gboolean requestCertificateCallback(SoupMessage*, GTlsClientConnection*, NetworkDataTaskSoup*);
    static gboolean requestCertificatePasswordCallback(SoupMessage*, GTlsPassword*, NetworkDataTaskSoup*);
#endif

    void download();
    static void writeDownloadCallback(GOutputStream*, GAsyncResult*, NetworkDataTaskSoup*);
    void writeDownload();
    void didWriteDownload(gsize bytesWritten);
    void didFailDownload(const WebCore::ResourceError&);
    void didFinishDownload();
    void cleanDownloadFiles();

    void didFail(const WebCore::ResourceError&);

#if USE(SOUP2)
    static void networkEventCallback(SoupMessage*, GSocketClientEvent, GIOStream*, NetworkDataTaskSoup*);
    void networkEvent(GSocketClientEvent, GIOStream*);
#endif

#if SOUP_CHECK_VERSION(2, 49, 91)
    static void startingCallback(SoupMessage*, NetworkDataTaskSoup*);
#else
    static void requestStartedCallback(SoupSession*, SoupMessage*, SoupSocket*, NetworkDataTaskSoup*);
#endif
#if SOUP_CHECK_VERSION(2, 67, 1)
    bool shouldAllowHSTSPolicySetting() const;
    bool shouldAllowHSTSProtocolUpgrade() const;
    void protocolUpgradedViaHSTS(SoupMessage*);
#if USE(SOUP2)
    static void hstsEnforced(SoupHSTSEnforcer*, SoupMessage*, NetworkDataTaskSoup*);
#else
    static void hstsEnforced(SoupMessage*, NetworkDataTaskSoup*);
#endif
#endif
    void didStartRequest();
    static void restartedCallback(SoupMessage*, NetworkDataTaskSoup*);
    void didRestart();

    static void fileQueryInfoCallback(GFile*, GAsyncResult*, NetworkDataTaskSoup*);
    void didGetFileInfo(GFileInfo*);
    static void readFileCallback(GFile*, GAsyncResult*, NetworkDataTaskSoup*);
    static void enumerateFileChildrenCallback(GFile*, GAsyncResult*, NetworkDataTaskSoup*);
    void didReadFile(GRefPtr<GInputStream>&&);

    WebCore::AdditionalNetworkLoadMetricsForWebInspector& additionalNetworkLoadMetricsForWebInspector();

    Markable<WebCore::FrameIdentifier> m_frameID;
    Markable<WebCore::PageIdentifier> m_pageID;
    State m_state { State::Suspended };
    WebCore::ContentSniffingPolicy m_shouldContentSniff;
    PreconnectOnly m_shouldPreconnectOnly { PreconnectOnly::No };
    GRefPtr<SoupMessage> m_soupMessage;
    GRefPtr<GFile> m_file;
    GRefPtr<GInputStream> m_inputStream;
    GRefPtr<SoupMultipartInputStream> m_multipartInputStream;
    GRefPtr<GCancellable> m_cancellable;
    GRefPtr<GAsyncResult> m_pendingResult;
    WebCore::ProtectionSpace m_protectionSpaceForPersistentStorage;
    WebCore::Credential m_credentialForPersistentStorage;
    WebCore::ResourceRequest m_currentRequest;
    WebCore::ResourceResponse m_response;
    CString m_sniffedContentType;
    Vector<uint8_t> m_readBuffer;
    uint64_t m_bodyDataTotalBytesSent { 0 };
    GRefPtr<GFile> m_downloadDestinationFile;
    GRefPtr<GFile> m_downloadIntermediateFile;
    GRefPtr<GOutputStream> m_downloadOutputStream;
    bool m_allowOverwriteDownload { false };
    WebCore::NetworkLoadMetrics m_networkLoadMetrics;
    bool m_isBlockingCookies { false };
    RefPtr<WebCore::SecurityOrigin> m_sourceOrigin;
    RunLoop::Timer m_timeoutSource;
};

} // namespace WebKit
