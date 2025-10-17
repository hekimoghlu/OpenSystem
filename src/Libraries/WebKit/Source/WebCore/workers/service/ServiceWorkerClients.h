/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 20, 2023.
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

#include "ServiceWorkerClientQueryOptions.h"
#include "ServiceWorkerClientType.h"
#include "ServiceWorkerIdentifier.h"
#include <wtf/HashMap.h>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>

namespace WebCore {

class DeferredPromise;
class ScriptExecutionContext;
class WebCoreOpaqueRoot;
struct ServiceWorkerClientData;

class ServiceWorkerClients : public RefCounted<ServiceWorkerClients> {
public:
    static Ref<ServiceWorkerClients> create()
    {
        return adoptRef(*new ServiceWorkerClients);
    }

    using ClientQueryOptions = ServiceWorkerClientQueryOptions;

    void get(ScriptExecutionContext&, const String& id, Ref<DeferredPromise>&&);
    void matchAll(ScriptExecutionContext&, const ClientQueryOptions&, Ref<DeferredPromise>&&);
    void openWindow(ScriptExecutionContext&, const String& url, Ref<DeferredPromise>&&);
    void claim(ScriptExecutionContext&, Ref<DeferredPromise>&&);

    enum PromiseIdentifierType { };
    using PromiseIdentifier = AtomicObjectIdentifier<PromiseIdentifierType>;

    PromiseIdentifier addPendingPromise(Ref<DeferredPromise>&&);
    RefPtr<DeferredPromise> takePendingPromise(PromiseIdentifier);

private:
    ServiceWorkerClients() = default;

    HashMap<PromiseIdentifier, Ref<DeferredPromise>> m_pendingPromises;
};

WebCoreOpaqueRoot root(ServiceWorkerClients*);

} // namespace WebCore
