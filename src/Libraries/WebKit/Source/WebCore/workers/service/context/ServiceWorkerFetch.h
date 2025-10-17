/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 18, 2025.
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

#include "FetchIdentifier.h"
#include "ResourceResponse.h"
#include "ScriptExecutionContextIdentifier.h"
#include "ServiceWorkerTypes.h"
#include <wtf/Ref.h>
#include <wtf/ThreadSafeRefCounted.h>

namespace WebCore {
class FetchEvent;
struct FetchOptions;
class FetchResponse;
class FormData;
class NetworkLoadMetrics;
class ResourceError;
class ResourceRequest;
class ServiceWorkerGlobalScope;
class ServiceWorkerGlobalScope;
class SharedBuffer;

namespace ServiceWorkerFetch {
class Client : public ThreadSafeRefCounted<Client, WTF::DestructionThread::Main> {
public:
    virtual ~Client() = default;

    virtual void didReceiveRedirection(const ResourceResponse&) = 0;
    virtual void didReceiveResponse(const ResourceResponse&) = 0;
    virtual void didReceiveData(const SharedBuffer&) = 0;
    virtual void didReceiveFormDataAndFinish(Ref<FormData>&&) = 0;
    virtual void didFail(const ResourceError&) = 0;
    virtual void didFinish(const NetworkLoadMetrics&) = 0;
    virtual void didNotHandle() = 0;
    virtual void setCancelledCallback(Function<void()>&&) = 0;
    virtual void usePreload() = 0;
    virtual void contextIsStopping() = 0;

    void cancel();
    bool isCancelled() const { return m_isCancelled; }

private:
    virtual void doCancel() = 0;
    bool m_isCancelled { false };
};

inline void Client::cancel()
{
    ASSERT(!m_isCancelled);
    m_isCancelled = true;
    doCancel();
}

void dispatchFetchEvent(Ref<Client>&&, ServiceWorkerGlobalScope&, ResourceRequest&&, String&& referrer, FetchOptions&&, SWServerConnectionIdentifier, FetchIdentifier, bool isServiceWorkerNavigationPreloadEnabled, String&& clientIdentifier, String&& resultingClientIdentifier);
};

} // namespace WebCore
