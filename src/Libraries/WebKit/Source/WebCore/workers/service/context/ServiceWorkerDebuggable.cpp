/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 7, 2022.
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
#include "config.h"
#include "ServiceWorkerDebuggable.h"

#if ENABLE(REMOTE_INSPECTOR)

#include "ServiceWorkerInspectorProxy.h"
#include "ServiceWorkerThreadProxy.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ServiceWorkerDebuggable);

using namespace Inspector;

Ref<ServiceWorkerDebuggable> ServiceWorkerDebuggable::create(ServiceWorkerThreadProxy& serviceWorkerThreadProxy, const ServiceWorkerContextData& data)
{
    return adoptRef(*new ServiceWorkerDebuggable(serviceWorkerThreadProxy, data));
}

ServiceWorkerDebuggable::ServiceWorkerDebuggable(ServiceWorkerThreadProxy& serviceWorkerThreadProxy, const ServiceWorkerContextData& data)
    : m_serviceWorkerThreadProxy(serviceWorkerThreadProxy)
    , m_scopeURL(data.registration.scopeURL.string())
{
}

void ServiceWorkerDebuggable::connect(FrontendChannel& channel, bool, bool)
{
    if (RefPtr serviceWorkerThreadProxy = m_serviceWorkerThreadProxy.get())
        serviceWorkerThreadProxy->inspectorProxy().connectToWorker(channel);
}

void ServiceWorkerDebuggable::disconnect(FrontendChannel& channel)
{
    if (RefPtr serviceWorkerThreadProxy = m_serviceWorkerThreadProxy.get())
        serviceWorkerThreadProxy->inspectorProxy().disconnectFromWorker(channel);
}

void ServiceWorkerDebuggable::dispatchMessageFromRemote(String&& message)
{
    if (RefPtr serviceWorkerThreadProxy = m_serviceWorkerThreadProxy.get())
        serviceWorkerThreadProxy->inspectorProxy().sendMessageToWorker(WTFMove(message));
}

} // namespace WebCore

#endif // ENABLE(REMOTE_INSPECTOR)
