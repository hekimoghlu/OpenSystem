/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 16, 2022.
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

#include "BackgroundFetchRecordLoader.h"
#include "ProcessIdentifier.h"
#include "ScriptExecutionContextIdentifier.h"
#include <wtf/CompletionHandler.h>
#include <wtf/HashSet.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class SWServerDelegate;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::SWServerDelegate> : std::true_type { };
}

namespace WebCore {

class BackgroundFetchEngine;
class BackgroundFetchRecordLoader;
class BackgroundFetchStore;
class RegistrableDomain;
class ResourceRequest;
class SWRegistrationStore;
class SWServer;
class Site;

struct BackgroundFetchRequest;
struct ServiceWorkerJobData;
struct WorkerFetchResult;

class SWServerDelegate : public CanMakeWeakPtr<SWServerDelegate> {
public:
    virtual ~SWServerDelegate() = default;

    virtual void softUpdate(ServiceWorkerJobData&&, bool shouldRefreshCache, ResourceRequest&&, CompletionHandler<void(WorkerFetchResult&&)>&&) = 0;
    virtual void createContextConnection(const Site&, std::optional<ProcessIdentifier>, std::optional<ScriptExecutionContextIdentifier>, CompletionHandler<void()>&&) = 0;
    virtual void appBoundDomains(CompletionHandler<void(HashSet<RegistrableDomain>&&)>&&) = 0;
    virtual void addAllowedFirstPartyForCookies(ProcessIdentifier, std::optional<ProcessIdentifier>, RegistrableDomain&&) = 0;

    virtual void requestBackgroundFetchPermission(const ClientOrigin&, CompletionHandler<void(bool)>&&) = 0;
    virtual RefPtr<BackgroundFetchRecordLoader> createBackgroundFetchRecordLoader(BackgroundFetchRecordLoaderClient&, const BackgroundFetchRequest&, size_t responseDataSize, const WebCore::ClientOrigin&) = 0;
    virtual Ref<BackgroundFetchStore> createBackgroundFetchStore() = 0;
    virtual RefPtr<SWRegistrationStore> createRegistrationStore(SWServer&) = 0;
};

} // namespace WebCore
