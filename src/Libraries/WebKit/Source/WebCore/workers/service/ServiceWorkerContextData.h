/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 14, 2023.
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

#include "CertificateInfo.h"
#include "ContentSecurityPolicyResponseHeaders.h"
#include "CrossOriginEmbedderPolicy.h"
#include "NavigationPreloadState.h"
#include "ScriptBuffer.h"
#include "ScriptExecutionContextIdentifier.h"
#include "ServiceWorkerIdentifier.h"
#include "ServiceWorkerImportedScript.h"
#include "ServiceWorkerJobDataIdentifier.h"
#include "ServiceWorkerRegistrationData.h"
#include "WorkerType.h"
#include <wtf/RobinHoodHashMap.h>
#include <wtf/URLHash.h>

namespace WebCore {

enum class LastNavigationWasAppInitiated : bool;

struct ServiceWorkerContextData {
    std::optional<ServiceWorkerJobDataIdentifier> jobDataIdentifier;
    ServiceWorkerRegistrationData registration;
    ServiceWorkerIdentifier serviceWorkerIdentifier;
    ScriptBuffer script;
    CertificateInfo certificateInfo;
    ContentSecurityPolicyResponseHeaders contentSecurityPolicy;
    CrossOriginEmbedderPolicy crossOriginEmbedderPolicy;
    String referrerPolicy;
    URL scriptURL;
    WorkerType workerType;
    bool loadedFromDisk;
    std::optional<LastNavigationWasAppInitiated> lastNavigationWasAppInitiated;
    MemoryCompactRobinHoodHashMap<URL, ServiceWorkerImportedScript> scriptResourceMap;
    std::optional<ScriptExecutionContextIdentifier> serviceWorkerPageIdentifier;
    NavigationPreloadState navigationPreloadState;
    
    using ImportedScript = ServiceWorkerImportedScript;

    WEBCORE_EXPORT ServiceWorkerContextData isolatedCopy() const &;
    WEBCORE_EXPORT ServiceWorkerContextData isolatedCopy() &&;
};

} // namespace WebCore
