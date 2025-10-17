/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 8, 2024.
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

#include "ScriptExecutionContextIdentifier.h"
#include "SecurityOriginData.h"
#include "ServiceWorkerJobDataIdentifier.h"
#include "ServiceWorkerJobType.h"
#include "ServiceWorkerRegistrationKey.h"
#include "ServiceWorkerRegistrationOptions.h"
#include "ServiceWorkerTypes.h"
#include <wtf/URL.h>

namespace WebCore {

struct ServiceWorkerJobData {
    using Identifier = ServiceWorkerJobDataIdentifier;
    ServiceWorkerJobData(SWServerConnectionIdentifier, const ServiceWorkerOrClientIdentifier& sourceContext);
    ServiceWorkerJobData(Identifier, const ServiceWorkerOrClientIdentifier& sourceContext);
    WEBCORE_EXPORT ServiceWorkerJobData(WebCore::ServiceWorkerJobDataIdentifier&&, URL&& scriptURL, URL&& clientCreationURL, WebCore::SecurityOriginData&& topOrigin, URL&& scopeURL, WebCore::ServiceWorkerOrClientIdentifier&& sourceContext, WebCore::WorkerType, WebCore::ServiceWorkerJobType, String&& domainForCachePartition, bool isFromServiceWorkerPage, std::optional<WebCore::ServiceWorkerRegistrationOptions>&&);

    SWServerConnectionIdentifier connectionIdentifier() const { return m_identifier.connectionIdentifier; }

    bool isEquivalent(const ServiceWorkerJobData&) const;
    std::optional<ScriptExecutionContextIdentifier> serviceWorkerPageIdentifier() const;

    URL scriptURL;
    URL clientCreationURL;
    SecurityOriginData topOrigin;
    URL scopeURL;
    ServiceWorkerOrClientIdentifier sourceContext;
    WorkerType workerType;
    ServiceWorkerJobType type;
    String domainForCachePartition;
    bool isFromServiceWorkerPage { false };

    std::optional<ServiceWorkerRegistrationOptions> registrationOptions;

    Identifier identifier() const { return m_identifier; }
    WEBCORE_EXPORT ServiceWorkerRegistrationKey registrationKey() const;
    ServiceWorkerJobData isolatedCopy() const;

private:
    Identifier m_identifier;
};

} // namespace WebCore
