/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 20, 2022.
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

#include "APIObject.h"
#include "Connection.h"
#include "DownloadID.h"
#include "IdentifierTypes.h"
#include "SandboxExtension.h"
#include "UseDownloadPlaceholder.h"
#include "WebsiteDataStore.h"
#include <WebCore/ResourceRequest.h>
#include <wtf/Forward.h>
#include <wtf/Ref.h>
#include <wtf/WeakPtr.h>

OBJC_CLASS NSProgress;

namespace API {
class Data;
class DownloadClient;
class FrameInfo;
}

namespace WebCore {
class AuthenticationChallenge;
class IntRect;
class ProtectionSpace;
class ResourceError;
class ResourceResponse;
}

namespace WebKit {

class DownloadProxyMap;
class ProcessAssertion;
class WebPageProxy;

enum class AllowOverwrite : bool;

struct FrameInfoData;

class DownloadProxy : public API::ObjectImpl<API::Object::Type::Download>, public IPC::MessageReceiver {
public:
    using DecideDestinationCallback = CompletionHandler<void(String, SandboxExtension::Handle, AllowOverwrite, WebKit::UseDownloadPlaceholder, const URL&, SandboxExtension::Handle, std::span<const uint8_t>, std::span<const uint8_t>)>;

    template<typename... Args> static Ref<DownloadProxy> create(Args&&... args)
    {
        return adoptRef(*new DownloadProxy(std::forward<Args>(args)...));
    }
    ~DownloadProxy();

    void ref() const final { API::ObjectImpl<API::Object::Type::Download>::ref(); }
    void deref() const final { API::ObjectImpl<API::Object::Type::Download>::deref(); }

    DownloadID downloadID() const { return m_downloadID; }
    const WebCore::ResourceRequest& request() const { return m_request; }
    API::Data* legacyResumeData() const { return m_legacyResumeData.get(); }

    void cancel(CompletionHandler<void(API::Data*)>&&);

    void invalidate();
    void processDidClose();

    void didReceiveDownloadProxyMessage(IPC::Connection&, IPC::Decoder&);
    bool didReceiveSyncDownloadProxyMessage(IPC::Connection&, IPC::Decoder&, UniqueRef<IPC::Encoder>&);

    WebPageProxy* originatingPage() const;

    void setRedirectChain(Vector<URL>&& redirectChain) { m_redirectChain = WTFMove(redirectChain); }
    const Vector<URL>& redirectChain() const { return m_redirectChain; }

    void setWasUserInitiated(bool value) { m_wasUserInitiated = value; }
    bool wasUserInitiated() const { return m_wasUserInitiated; }

    const String& destinationFilename() const { return m_destinationFilename; }
    void setDestinationFilename(const String& d) { m_destinationFilename = d; }

#if PLATFORM(COCOA)
    void publishProgress(const URL&);
    void setProgress(NSProgress *progress) { m_progress = progress; }
    NSProgress *progress() const { return m_progress.get(); }
#endif
#if PLATFORM(MAC)
    void updateQuarantinePropertiesIfPossible();
#endif
    API::FrameInfo& frameInfo() { return m_frameInfo.get(); }

    API::DownloadClient& client() { return m_client.get(); }
    void setClient(Ref<API::DownloadClient>&&);
    void setDidStartCallback(CompletionHandler<void(DownloadProxy*)>&& callback) { m_didStartCallback = WTFMove(callback); }
    void setSuggestedFilename(const String& suggestedFilename) { m_suggestedFilename = suggestedFilename; }

    // Message handlers.
    void didStart(const WebCore::ResourceRequest&, const String& suggestedFilename);
    void didReceiveAuthenticationChallenge(WebCore::AuthenticationChallenge&&, AuthenticationChallengeIdentifier);
    void didReceiveData(uint64_t bytesWritten, uint64_t totalBytesWritten, uint64_t totalBytesExpectedToWrite);
    void shouldDecodeSourceDataOfMIMEType(const String& mimeType, bool& result);
    void didCreateDestination(const String& path);
    void didFinish();
    void didFail(const WebCore::ResourceError&, std::span<const uint8_t> resumeData);
#if HAVE(MODERN_DOWNLOADPROGRESS)
    void didReceivePlaceholderURL(const URL&, std::span<const uint8_t> bookmarkData, WebKit::SandboxExtensionHandle&&, CompletionHandler<void()>&&);
    void didReceiveFinalURL(const URL&, std::span<const uint8_t> bookmarkData, WebKit::SandboxExtensionHandle&&);
    void didStartUpdatingProgress();
#endif
    void willSendRequest(WebCore::ResourceRequest&& redirectRequest, const WebCore::ResourceResponse& redirectResponse, CompletionHandler<void(WebCore::ResourceRequest&&)>&&);
    void decideDestinationWithSuggestedFilename(const WebCore::ResourceResponse&, String&& suggestedFilename, DecideDestinationCallback&&);

private:
    explicit DownloadProxy(DownloadProxyMap&, WebsiteDataStore&, API::DownloadClient&, const WebCore::ResourceRequest&, const FrameInfoData&, WebPageProxy*);

    Ref<API::DownloadClient> protectedClient() const;
    RefPtr<WebsiteDataStore> protectedDataStore() { return m_dataStore; }

    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) override;

#if HAVE(MODERN_DOWNLOADPROGRESS)
    static Vector<uint8_t> bookmarkDataForURL(const URL&);
    static Vector<uint8_t> activityAccessToken();
#endif

    WeakPtr<DownloadProxyMap> m_downloadProxyMap;
    RefPtr<WebsiteDataStore> m_dataStore;
    Ref<API::DownloadClient> m_client;
    DownloadID m_downloadID;

    RefPtr<API::Data> m_legacyResumeData;
    WebCore::ResourceRequest m_request;
    String m_suggestedFilename;
    String m_destinationFilename;

    WeakPtr<WebPageProxy> m_originatingPage;
    Vector<URL> m_redirectChain;
    bool m_wasUserInitiated { true };
    bool m_downloadIsCancelled { false };
    Ref<API::FrameInfo> m_frameInfo;
    CompletionHandler<void(DownloadProxy*)> m_didStartCallback;
#if PLATFORM(COCOA)
    RetainPtr<NSProgress> m_progress;
#endif
#if HAVE(MODERN_DOWNLOADPROGRESS)
    RefPtr<ProcessAssertion> m_assertion;
#endif
};

} // namespace WebKit
