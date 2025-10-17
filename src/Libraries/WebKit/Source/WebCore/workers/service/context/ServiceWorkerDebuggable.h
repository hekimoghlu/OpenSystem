/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 8, 2022.
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

#if ENABLE(REMOTE_INSPECTOR)

#include "ServiceWorkerContextData.h"
#include <JavaScriptCore/RemoteInspectionTarget.h>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class ServiceWorkerThreadProxy;

class ServiceWorkerDebuggable final : public Inspector::RemoteInspectionTarget {
    WTF_MAKE_TZONE_ALLOCATED(ServiceWorkerDebuggable);
    WTF_MAKE_NONCOPYABLE(ServiceWorkerDebuggable);
public:
    static Ref<ServiceWorkerDebuggable> create(ServiceWorkerThreadProxy&, const ServiceWorkerContextData&);
    ~ServiceWorkerDebuggable() = default;

    Inspector::RemoteControllableTarget::Type type() const final { return Inspector::RemoteControllableTarget::Type::ServiceWorker; }

    String name() const final { return "ServiceWorker"_s; }
    String url() const final { return m_scopeURL; }
    bool hasLocalDebugger() const final { return false; }

    void connect(Inspector::FrontendChannel&, bool isAutomaticConnection = false, bool immediatelyPause = false) final;
    void disconnect(Inspector::FrontendChannel&) final;
    void dispatchMessageFromRemote(String&& message) final;

private:
    ServiceWorkerDebuggable(ServiceWorkerThreadProxy&, const ServiceWorkerContextData&);

    ThreadSafeWeakPtr<ServiceWorkerThreadProxy> m_serviceWorkerThreadProxy;
    String m_scopeURL;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CONTROLLABLE_TARGET(WebCore::ServiceWorkerDebuggable, ServiceWorker);

#endif // ENABLE(REMOTE_INSPECTOR)
