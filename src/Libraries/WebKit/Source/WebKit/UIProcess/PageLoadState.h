/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 30, 2022.
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

#include <WebCore/CertificateInfo.h>
#include <WebCore/NavigationIdentifier.h>
#include <WebCore/SecurityOriginData.h>
#include <wtf/AbstractRefCountedAndCanMakeWeakPtr.h>
#include <wtf/URL.h>
#include <wtf/WeakHashSet.h>
#include <wtf/text/WTFString.h>

namespace WebCore {
enum class ResourceResponseSource : uint8_t;
}
namespace WebKit {

class WebPageProxy;
enum class Source;

class PageLoadStateObserverBase : public AbstractRefCountedAndCanMakeWeakPtr<PageLoadStateObserverBase> {
public:
    virtual ~PageLoadStateObserverBase() = default;

    virtual void willChangeIsLoading() = 0;
    virtual void didChangeIsLoading() = 0;

    virtual void willChangeTitle() = 0;
    virtual void didChangeTitle() = 0;

    virtual void willChangeActiveURL() = 0;
    virtual void didChangeActiveURL() = 0;

    virtual void willChangeHasOnlySecureContent() = 0;
    virtual void didChangeHasOnlySecureContent() = 0;

    virtual void willChangeNegotiatedLegacyTLS() { }
    virtual void didChangeNegotiatedLegacyTLS() { }

    virtual void willChangeWasPrivateRelayed() { }
    virtual void didChangeWasPrivateRelayed() { }

    virtual void willChangeEstimatedProgress() = 0;
    virtual void didChangeEstimatedProgress() = 0;

    virtual void willChangeCanGoBack() = 0;
    virtual void didChangeCanGoBack() = 0;

    virtual void willChangeCanGoForward() = 0;
    virtual void didChangeCanGoForward() = 0;

    virtual void willChangeNetworkRequestsInProgress() = 0;
    virtual void didChangeNetworkRequestsInProgress() = 0;

    virtual void willChangeCertificateInfo() = 0;
    virtual void didChangeCertificateInfo() = 0;

    virtual void willChangeWebProcessIsResponsive() = 0;
    virtual void didChangeWebProcessIsResponsive() = 0;

    virtual void didSwapWebProcesses() = 0;
};

class PageLoadState {
public:
    explicit PageLoadState(WebPageProxy&);
    ~PageLoadState();

    enum class State : uint8_t { Provisional, Committed, Finished };

    using Observer = PageLoadStateObserverBase;

    class Transaction {
        WTF_MAKE_NONCOPYABLE(Transaction);
    public:
        Transaction(Transaction&&);
        ~Transaction();

    private:
        friend class PageLoadState;

        explicit Transaction(PageLoadState&);

        class Token {
        public:
            Token(Transaction& transaction)
#if ASSERT_ENABLED
                : m_pageLoadState(*transaction.m_pageLoadState)
#endif
            {
                transaction.m_pageLoadState->m_mayHaveUncommittedChanges = true;
            }

#if ASSERT_ENABLED
            PageLoadState& m_pageLoadState;
#endif
        };

        RefPtr<PageLoadState> m_pageLoadState;
    };

    struct PendingAPIRequest {
        Markable<WebCore::NavigationIdentifier> navigationID;
        String url;
    };

    void ref() const;
    void deref() const;

    void addObserver(Observer&);
    void removeObserver(Observer&);

    Transaction transaction() { return Transaction(*this); }
    void commitChanges();

    void reset(const Transaction::Token&);

    bool isLoading() const { return isLoading(m_committedState); }
    bool isProvisional() const { return m_committedState.state == State::Provisional; }
    bool isCommitted() const { return m_committedState.state == State::Committed; }
    bool isFinished() const { return m_committedState.state == State::Finished; }

    bool hasUncommittedLoad() const { return isLoading(m_uncommittedState); }

    const String& provisionalURL() const { return m_committedState.provisionalURL; }
    const String& url() const { return m_committedState.url; }
    const WebCore::SecurityOriginData& origin() const { return m_committedState.origin; }
    const String& unreachableURL() const { return m_committedState.unreachableURL; }

    String activeURL() const { return activeURL(m_committedState); }

    bool hasOnlySecureContent() const;
    bool hasNegotiatedLegacyTLS() const;
    void negotiatedLegacyTLS(const Transaction::Token&);
    bool wasPrivateRelayed() const { return m_committedState.wasPrivateRelayed; }
    String proxyName() { return m_committedState.proxyName; }
    WebCore::ResourceResponseSource source() { return m_committedState.source; }

    double estimatedProgress() const;
    bool networkRequestsInProgress() const { return m_committedState.networkRequestsInProgress; }

    const WebCore::CertificateInfo& certificateInfo() const { return m_committedState.certificateInfo; }

    const URL& resourceDirectoryURL() const { return m_committedState.resourceDirectoryURL; }

    const String& pendingAPIRequestURL() const { return m_committedState.pendingAPIRequest.url; }
    const PendingAPIRequest& pendingAPIRequest() const { return m_committedState.pendingAPIRequest; }
    void setPendingAPIRequest(const Transaction::Token&, PendingAPIRequest&& pendingAPIRequest, const URL& resourceDirectoryPath = { });
    void clearPendingAPIRequest(const Transaction::Token&);

    void didStartProvisionalLoad(const Transaction::Token&, const String& url, const String& unreachableURL);
    void didExplicitOpen(const Transaction::Token&, const String& url);
    void didReceiveServerRedirectForProvisionalLoad(const Transaction::Token&, const String& url);
    void didFailProvisionalLoad(const Transaction::Token&);

    void didCommitLoad(const Transaction::Token&, const WebCore::CertificateInfo&, bool hasInsecureContent, bool usedLegacyTLS, bool privateRelayed, const String& proxyName, const WebCore::ResourceResponseSource, const WebCore::SecurityOriginData&);

    void didFinishLoad(const Transaction::Token&);
    void didFailLoad(const Transaction::Token&);

    void didSameDocumentNavigation(const Transaction::Token&, const String& url);

    void didDisplayOrRunInsecureContent(const Transaction::Token&);

    void setUnreachableURL(const Transaction::Token&, const String&);

    const String& title() const;
    void setTitle(const Transaction::Token&, const String&);
    void setTitleFromBrowsingWarning(const Transaction::Token&, const String&);

    bool canGoBack() const;
    void setCanGoBack(const Transaction::Token&, bool);

    bool canGoForward() const;
    void setCanGoForward(const Transaction::Token&, bool);

    void didStartProgress(const Transaction::Token&);
    void didChangeProgress(const Transaction::Token&, double);
    void didFinishProgress(const Transaction::Token&);
    void setNetworkRequestsInProgress(const Transaction::Token&, bool);
    void setHTTPFallbackInProgress(const Transaction::Token&, bool);
    bool httpFallbackInProgress();

    void didSwapWebProcesses();

    bool committedHasInsecureContent() const { return m_committedState.hasInsecureContent; }

    // FIXME: We piggy-back off PageLoadState::Observer so that both WKWebView and WKObservablePageState
    // can listen for changes. Once we get rid of WKObservablePageState these could just be part of API::NavigationClient.
    void willChangeProcessIsResponsive();
    void didChangeProcessIsResponsive();

private:
    void beginTransaction() { ++m_outstandingTransactionCount; }
    void endTransaction();

    void callObserverCallback(void (Observer::*)());

    WeakHashSet<Observer> m_observers;

    struct Data {
        State state { State::Finished };
        bool hasInsecureContent { false };
        bool negotiatedLegacyTLS { false };
        bool wasPrivateRelayed { false };

        PendingAPIRequest pendingAPIRequest;

        String provisionalURL;
        String url;
        WebCore::SecurityOriginData origin;

        String unreachableURL;

        String title;
        String titleFromBrowsingWarning;

        URL resourceDirectoryURL;

        bool canGoBack { false };
        bool canGoForward { false };
        bool isHTTPFallbackInProgress { false };

        double estimatedProgress { 0 };
        bool networkRequestsInProgress { false };

        WebCore::CertificateInfo certificateInfo;
        String proxyName;
        WebCore::ResourceResponseSource source;
    };

    static bool isLoading(const Data&);
    static String activeURL(const Data&);
    static bool hasOnlySecureContent(const Data&);
    static double estimatedProgress(const Data&);

    Ref<WebPageProxy> protectedPage() const;

    WeakRef<WebPageProxy> m_webPageProxy;

    Data m_committedState;
    Data m_uncommittedState;

    String m_lastUnreachableURL;

    bool m_mayHaveUncommittedChanges;
    unsigned m_outstandingTransactionCount;
};

} // namespace WebKit
