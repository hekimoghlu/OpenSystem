/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 21, 2022.
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

#include "AdvancedPrivacyProtections.h"
#include "CertificateInfo.h"
#include "ContentSecurityPolicyResponseHeaders.h"
#include "CrossOriginEmbedderPolicy.h"
#include "FetchOptions.h"
#include "ResourceError.h"
#include "ResourceRequest.h"
#include "ResourceResponse.h"
#include "ScriptBuffer.h"
#include "ScriptExecutionContextIdentifier.h"
#include "ServiceWorkerRegistrationData.h"
#include "ThreadableLoader.h"
#include "ThreadableLoaderClient.h"
#include <memory>
#include <wtf/OptionSet.h>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/URL.h>
#include <wtf/text/StringBuilder.h>

namespace WebCore {

class Exception;
class ScriptExecutionContext;
class TextResourceDecoder;
class WorkerScriptLoaderClient;
struct ServiceWorkerRegistrationData;
struct WorkerFetchResult;
enum class CertificateInfoPolicy : uint8_t;

class WorkerScriptLoader final : public RefCounted<WorkerScriptLoader>, public ThreadableLoaderClient {
    WTF_MAKE_TZONE_ALLOCATED(WorkerScriptLoader);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(WorkerScriptLoader);
public:
    static Ref<WorkerScriptLoader> create()
    {
        return adoptRef(*new WorkerScriptLoader);
    }

    enum class Source : uint8_t { ClassicWorkerScript, ClassicWorkerImport, ModuleScript };

    std::optional<Exception> loadSynchronously(ScriptExecutionContext*, const URL&, Source, FetchOptions::Mode, FetchOptions::Cache, ContentSecurityPolicyEnforcement, const String& initiatorIdentifier);
    void loadAsynchronously(ScriptExecutionContext&, ResourceRequest&&, Source, FetchOptions&&, ContentSecurityPolicyEnforcement, ServiceWorkersMode, WorkerScriptLoaderClient&, String&& taskMode, std::optional<ScriptExecutionContextIdentifier> clientIdentifier = std::nullopt);

    void notifyError(std::optional<ScriptExecutionContextIdentifier>);

    OptionSet<AdvancedPrivacyProtections> advancedPrivacyProtections() const { return m_advancedPrivacyProtections; }

    const ScriptBuffer& script() const { return m_script; }
    const ContentSecurityPolicyResponseHeaders& contentSecurityPolicy() const { return m_contentSecurityPolicy; }
    const String& referrerPolicy() const { return m_referrerPolicy; }
    const CrossOriginEmbedderPolicy& crossOriginEmbedderPolicy() const { return m_crossOriginEmbedderPolicy; }
    const URL& url() const { return m_url; }
    const URL& responseURL() const;
    ResourceResponse::Source responseSource() const { return m_responseSource; }
    bool isRedirected() const { return m_isRedirected; }
    const CertificateInfo& certificateInfo() const { return m_certificateInfo; }
    const String& responseMIMEType() const { return m_responseMIMEType; }
    ResourceResponse::Tainting responseTainting() const { return m_responseTainting; }
    bool failed() const { return m_failed; }
    ResourceLoaderIdentifier identifier() const { return *m_identifier; }
    const ResourceError& error() const { return m_error; }

    WorkerFetchResult fetchResult() const;

    void didReceiveResponse(ScriptExecutionContextIdentifier, std::optional<ResourceLoaderIdentifier>, const ResourceResponse&) override;
    void didReceiveData(const SharedBuffer&) override;
    void didFinishLoading(ScriptExecutionContextIdentifier, std::optional<ResourceLoaderIdentifier>, const NetworkLoadMetrics&) override;
    void didFail(std::optional<ScriptExecutionContextIdentifier>, const ResourceError&) override;

    void cancel();

    WEBCORE_EXPORT static ResourceError validateWorkerResponse(const ResourceResponse&, Source, FetchOptions::Destination);

    class ServiceWorkerDataManager : public ThreadSafeRefCounted<ServiceWorkerDataManager, WTF::DestructionThread::Main> {
    public:
        static Ref<ServiceWorkerDataManager> create(ScriptExecutionContextIdentifier identifier) { return adoptRef(*new ServiceWorkerDataManager(identifier)); }
        WEBCORE_EXPORT ~ServiceWorkerDataManager();

        WEBCORE_EXPORT void setData(ServiceWorkerData&&);
        std::optional<ServiceWorkerData> takeData();

    private:
        explicit ServiceWorkerDataManager(ScriptExecutionContextIdentifier identifier)
            : m_clientIdentifier(identifier)
        {
        }

        ScriptExecutionContextIdentifier m_clientIdentifier;
        Lock m_activeServiceWorkerDataLock;
        std::optional<ServiceWorkerData> m_activeServiceWorkerData WTF_GUARDED_BY_LOCK(m_activeServiceWorkerDataLock);
    };

    void setControllingServiceWorker(ServiceWorkerData&&);
    std::optional<ServiceWorkerData> takeServiceWorkerData();
    WEBCORE_EXPORT static RefPtr<ServiceWorkerDataManager> serviceWorkerDataManagerFromIdentifier(ScriptExecutionContextIdentifier);

    std::optional<ScriptExecutionContextIdentifier> clientIdentifier() const { return m_clientIdentifier; }
    const String& userAgentForSharedWorker() const { return m_userAgentForSharedWorker; }

private:
    friend class RefCounted<WorkerScriptLoader>;
    friend struct std::default_delete<WorkerScriptLoader>;

    WorkerScriptLoader();
    ~WorkerScriptLoader();

    std::unique_ptr<ResourceRequest> createResourceRequest(const String& initiatorIdentifier);
    void notifyFinished(std::optional<ScriptExecutionContextIdentifier>);

    WeakPtr<WorkerScriptLoaderClient> m_client;
    RefPtr<ThreadableLoader> m_threadableLoader;
    RefPtr<TextResourceDecoder> m_decoder;
    ScriptBuffer m_script;
    URL m_url;
    URL m_responseURL;
    CertificateInfo m_certificateInfo;
    String m_responseMIMEType;
    Source m_source;
    FetchOptions::Destination m_destination;
    ContentSecurityPolicyResponseHeaders m_contentSecurityPolicy;
    String m_referrerPolicy;
    CrossOriginEmbedderPolicy m_crossOriginEmbedderPolicy;
    Markable<ResourceLoaderIdentifier> m_identifier;
    bool m_failed { false };
    bool m_finishing { false };
    bool m_isRedirected { false };
    bool m_isCOEPEnabled { false };
    ResourceResponse::Source m_responseSource { ResourceResponse::Source::Unknown };
    ResourceResponse::Tainting m_responseTainting { ResourceResponse::Tainting::Basic };
    ResourceError m_error;
    Markable<ScriptExecutionContextIdentifier> m_clientIdentifier;
    bool m_didAddToWorkerScriptLoaderMap { false };
    bool m_isMatchingServiceWorkerRegistration { false };
    std::optional<SecurityOriginData> m_topOriginForServiceWorkerRegistration;
    RefPtr<ServiceWorkerDataManager> m_serviceWorkerDataManager;
    WeakPtr<ScriptExecutionContext> m_context;
    String m_userAgentForSharedWorker;
    OptionSet<AdvancedPrivacyProtections> m_advancedPrivacyProtections;
};

} // namespace WebCore
