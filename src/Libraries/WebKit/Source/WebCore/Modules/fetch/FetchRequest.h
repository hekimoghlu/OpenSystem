/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 18, 2024.
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

#include "AbortSignal.h"
#include "ExceptionOr.h"
#include "FetchBodyOwner.h"
#include "FetchIdentifier.h"
#include "FetchOptions.h"
#include "FetchRequestDestination.h"
#include "FetchRequestInit.h"
#include "ResourceRequest.h"
#include "URLKeepingBlobAlive.h"

namespace WebCore {

class Blob;
class ScriptExecutionContext;
class URLSearchParams;
class WebCoreOpaqueRoot;

class FetchRequest final : public FetchBodyOwner {
public:
    using Init = FetchRequestInit;
    using Info = std::variant<RefPtr<FetchRequest>, String>;

    using Cache = FetchOptions::Cache;
    using Credentials = FetchOptions::Credentials;
    using Destination = FetchOptions::Destination;
    using Mode = FetchOptions::Mode;
    using Redirect = FetchOptions::Redirect;

    static ExceptionOr<Ref<FetchRequest>> create(ScriptExecutionContext&, Info&&, Init&&);
    static Ref<FetchRequest> create(ScriptExecutionContext&, std::optional<FetchBody>&&, Ref<FetchHeaders>&&, ResourceRequest&&, FetchOptions&&, String&& referrer);

    const String& method() const { return m_request.httpMethod(); }
    const String& urlString() const;
    FetchHeaders& headers() { return m_headers.get(); }
    const FetchHeaders& headers() const { return m_headers.get(); }

    Destination destination() const { return m_options.destination; }
    String referrer() const;
    ReferrerPolicy referrerPolicy() const { return m_options.referrerPolicy; }
    Mode mode() const { return m_options.mode; }
    Credentials credentials() const { return m_options.credentials; }
    Cache cache() const { return m_options.cache; }
    Redirect redirect() const { return m_options.redirect; }
    bool keepalive() const { return m_options.keepAlive; };
    AbortSignal& signal() { return m_signal.get(); }

    const String& integrity() const { return m_options.integrity; }

    ExceptionOr<Ref<FetchRequest>> clone();

    const FetchOptions& fetchOptions() const { return m_options; }
    const ResourceRequest& internalRequest() const { return m_request; }
    const String& internalRequestReferrer() const { return m_referrer; }
    const URL& url() const { return m_request.url(); }

    ResourceRequest resourceRequest() const;
    std::optional<FetchIdentifier> navigationPreloadIdentifier() const { return m_navigationPreloadIdentifier.asOptional(); }
    void setNavigationPreloadIdentifier(std::optional<FetchIdentifier> identifier) { m_navigationPreloadIdentifier = identifier; }

    RequestPriority priority() const { return m_priority; }

    bool shouldEnableContentExtensionsCheck() const { return m_enableContentExtensionsCheck; }
    void disableContentExtensionsCheck() { m_enableContentExtensionsCheck = false; }

private:
    FetchRequest(ScriptExecutionContext&, std::optional<FetchBody>&&, Ref<FetchHeaders>&&, ResourceRequest&&, FetchOptions&&, String&& referrer);

    ExceptionOr<void> initializeOptions(const Init&);
    ExceptionOr<void> initializeWith(FetchRequest&, Init&&);
    ExceptionOr<void> initializeWith(const String&, Init&&);
    ExceptionOr<void> setBody(FetchBody::Init&&);
    ExceptionOr<void> setBody(FetchRequest&);

    void stop() final;

    Ref<AbortSignal> protectedSignal() const { return m_signal; }

    ResourceRequest m_request;
    URLKeepingBlobAlive m_requestURL;
    FetchOptions m_options;
    RequestPriority m_priority { RequestPriority::Auto };
    String m_referrer;
    Ref<AbortSignal> m_signal;
    Markable<FetchIdentifier> m_navigationPreloadIdentifier;
    bool m_enableContentExtensionsCheck { true };
};

WebCoreOpaqueRoot root(FetchRequest*);

} // namespace WebCore
