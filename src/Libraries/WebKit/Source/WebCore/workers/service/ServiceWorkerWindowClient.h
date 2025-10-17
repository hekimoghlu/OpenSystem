/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 7, 2022.
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

#include "ServiceWorkerClient.h"
#include "VisibilityState.h"

namespace WebCore {

class DeferredPromise;
class ServiceWorkerGlobalScope;

class ServiceWorkerWindowClient final : public ServiceWorkerClient {
public:
    static Ref<ServiceWorkerWindowClient> create(ServiceWorkerGlobalScope& context, ServiceWorkerClientData&& data)
    {
        return adoptRef(*new ServiceWorkerWindowClient(context, WTFMove(data)));
    }

    VisibilityState visibilityState() const { return data().isVisible ? VisibilityState::Visible : VisibilityState::Hidden; }
    bool focused() const { return data().isFocused; }
    const Vector<String>& ancestorOrigins() const { return data().ancestorOrigins; }

    void focus(ScriptExecutionContext&, Ref<DeferredPromise>&&);
    void navigate(ScriptExecutionContext&, const String& url, Ref<DeferredPromise>&&);

private:
    ServiceWorkerWindowClient(ServiceWorkerGlobalScope&, ServiceWorkerClientData&&);
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::ServiceWorkerWindowClient)
    static bool isType(const WebCore::ServiceWorkerClient& client) { return client.type() == WebCore::ServiceWorkerClientType::Window; }
SPECIALIZE_TYPE_TRAITS_END()
