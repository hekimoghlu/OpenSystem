/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 12, 2025.
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

#include "ContentSecurityPolicy.h"
#include "CrossOriginPreflightChecker.h"
#include "LoaderMalloc.h"
#include "ResourceLoaderIdentifier.h"
#include "ResourceResponse.h"
#include "SecurityOrigin.h"
#include "ThreadableLoader.h"
#include <wtf/WeakPtr.h>

namespace WebCore {
class CachedRawResource;
    class ContentSecurityPolicy;
    class Document;
    class ThreadableLoaderClient;
    class WeakPtrImplWithEventTargetData;

    class DocumentThreadableLoader : public RefCounted<DocumentThreadableLoader>, public ThreadableLoader, public CachedRawResourceClient {
        WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(Loader);
    public:
        static void loadResourceSynchronously(Document&, ResourceRequest&&, ThreadableLoaderClient&, const ThreadableLoaderOptions&, RefPtr<SecurityOrigin>&&, std::unique_ptr<ContentSecurityPolicy>&&, std::optional<CrossOriginEmbedderPolicy>&&);
        static void loadResourceSynchronously(Document&, ResourceRequest&&, ThreadableLoaderClient&, const ThreadableLoaderOptions&);

        enum class ShouldLogError : bool { No, Yes };
        static RefPtr<DocumentThreadableLoader> create(Document&, ThreadableLoaderClient&, ResourceRequest&&, const ThreadableLoaderOptions&, RefPtr<SecurityOrigin>&&, std::unique_ptr<ContentSecurityPolicy>&&, std::optional<CrossOriginEmbedderPolicy>&&, String&& referrer, ShouldLogError);
        static RefPtr<DocumentThreadableLoader> create(Document&, ThreadableLoaderClient&, ResourceRequest&&, const ThreadableLoaderOptions&, String&& referrer = String());

        virtual ~DocumentThreadableLoader();

        void cancel() override;
        virtual void setDefersLoading(bool);
        void computeIsDone() final;
        void clearClient() { m_client = nullptr; }

        friend CrossOriginPreflightChecker;
        friend class InspectorInstrumentation;
        friend class InspectorNetworkAgent;

        using RefCounted<DocumentThreadableLoader>::ref;
        using RefCounted<DocumentThreadableLoader>::deref;

    protected:
        void refThreadableLoader() override { ref(); }
        void derefThreadableLoader() override { deref(); }

    private:
        enum BlockingBehavior {
            LoadSynchronously,
            LoadAsynchronously
        };

        DocumentThreadableLoader(Document&, ThreadableLoaderClient&, BlockingBehavior, ResourceRequest&&, const ThreadableLoaderOptions&, RefPtr<SecurityOrigin>&&, std::unique_ptr<ContentSecurityPolicy>&&, std::optional<CrossOriginEmbedderPolicy>&&, String&&, ShouldLogError);

        void clearResource();

        // CachedRawResourceClient
        void dataSent(CachedResource&, unsigned long long bytesSent, unsigned long long totalBytesToBeSent) override;
        void responseReceived(CachedResource&, const ResourceResponse&, CompletionHandler<void()>&&) override;
        void dataReceived(CachedResource&, const SharedBuffer&) override;
        void redirectReceived(CachedResource&, ResourceRequest&&, const ResourceResponse&, CompletionHandler<void(ResourceRequest&&)>&&) override;
        void finishedTimingForWorkerLoad(CachedResource&, const ResourceTiming&) override;
        void finishedTimingForWorkerLoad(const ResourceTiming&);
        void notifyFinished(CachedResource&, const NetworkLoadMetrics&, LoadWillContinueInAnotherProcess) override;

        void didReceiveResponse(ResourceLoaderIdentifier, const ResourceResponse&);
        void didReceiveData(const SharedBuffer&);
        void didFinishLoading(std::optional<ResourceLoaderIdentifier>, const NetworkLoadMetrics&);
        void didFail(std::optional<ResourceLoaderIdentifier>, const ResourceError&);
        void makeCrossOriginAccessRequest(ResourceRequest&&);
        void makeSimpleCrossOriginAccessRequest(ResourceRequest&&);
        void makeCrossOriginAccessRequestWithPreflight(ResourceRequest&&);
        void preflightSuccess(ResourceRequest&&);
        void preflightFailure(std::optional<ResourceLoaderIdentifier>, const ResourceError&);

        void loadRequest(ResourceRequest&&, SecurityCheckPolicy);
        bool isAllowedRedirect(const URL&);
        bool isAllowedByContentSecurityPolicy(const URL&, ContentSecurityPolicy::RedirectResponseReceived, const URL& preRedirectURL = URL());
        bool isResponseAllowedByContentSecurityPolicy(const ResourceResponse&);

        Ref<SecurityOrigin> topOrigin() const;
        SecurityOrigin& securityOrigin() const;
        Ref<SecurityOrigin> protectedSecurityOrigin() const;
        const ContentSecurityPolicy& contentSecurityPolicy() const;
        CheckedRef<const ContentSecurityPolicy> checkedContentSecurityPolicy() const;
        const CrossOriginEmbedderPolicy& crossOriginEmbedderPolicy() const;

        Document& document() { return *m_document; }
        Ref<Document> protectedDocument();

        const ThreadableLoaderOptions& options() const { return m_options; }
        const String& referrer() const { return m_referrer; }
        bool isLoading() { return m_resource || m_preflightChecker; }

        void reportRedirectionWithBadScheme(const URL&);
        void reportContentSecurityPolicyError(const URL&);
        void reportCrossOriginResourceSharingError(const URL&);
        void reportIntegrityMetadataError(const CachedResource&, const String& expectedMetadata);
        void logErrorAndFail(const ResourceError&);

        bool shouldSetHTTPHeadersToKeep() const;
        bool checkURLSchemeAsCORSEnabled(const URL&);

        CachedResourceHandle<CachedRawResource> protectedResource() const;

        CachedResourceHandle<CachedRawResource> m_resource;
        ThreadableLoaderClient* m_client; // FIXME: Use a smart pointer.
        WeakPtr<Document, WeakPtrImplWithEventTargetData> m_document;
        ThreadableLoaderOptions m_options;
        bool m_responsesCanBeOpaque { true };
        RefPtr<SecurityOrigin> m_origin;
        String m_referrer;
        bool m_sameOriginRequest;
        bool m_simpleRequest;
        bool m_async;
        bool m_delayCallbacksForIntegrityCheck;
        std::unique_ptr<ContentSecurityPolicy> m_contentSecurityPolicy;
        std::optional<CrossOriginEmbedderPolicy> m_crossOriginEmbedderPolicy;
        std::optional<CrossOriginPreflightChecker> m_preflightChecker;
        std::optional<HTTPHeaderMap> m_originalHeaders;
        URL m_responseURL;

        ShouldLogError m_shouldLogError;
        std::optional<ResourceRequest> m_bypassingPreflightForServiceWorkerRequest;
    };

} // namespace WebCore
