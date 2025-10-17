/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 15, 2021.
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
#include "WorkerNavigator.h"

#include "Chrome.h"
#include "GPU.h"
#include "JSDOMPromiseDeferred.h"
#include "Page.h"
#include "PushEvent.h"
#include "ServiceWorkerGlobalScope.h"
#include "WorkerBadgeProxy.h"
#include "WorkerGlobalScope.h"
#include "WorkerThread.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(WorkerNavigator);

WorkerNavigator::WorkerNavigator(ScriptExecutionContext& context, const String& userAgent, bool isOnline)
    : NavigatorBase(&context)
    , m_userAgent(userAgent)
    , m_isOnline(isOnline)
{
}

WorkerNavigator::~WorkerNavigator() = default;

const String& WorkerNavigator::userAgent() const
{
    return m_userAgent;
}

bool WorkerNavigator::onLine() const
{
    return m_isOnline;
}

GPU* WorkerNavigator::gpu()
{
#if HAVE(WEBGPU_IMPLEMENTATION)
    if (!m_gpuForWebGPU) {
        auto scriptExecutionContext = this->scriptExecutionContext();
        if (scriptExecutionContext->isWorkerGlobalScope()) {
            WorkerGlobalScope& workerGlobalScope = downcast<WorkerGlobalScope>(*scriptExecutionContext);
            if (!workerGlobalScope.graphicsClient())
                return nullptr;

            RefPtr gpu = workerGlobalScope.graphicsClient()->createGPUForWebGPU();
            if (!gpu)
                return nullptr;

            m_gpuForWebGPU = GPU::create(*gpu);
        } else if (scriptExecutionContext->isDocument()) {
            Ref document = downcast<Document>(*scriptExecutionContext);
            RefPtr page = document->page();
            if (!page)
                return nullptr;
            RefPtr gpu = page->chrome().createGPUForWebGPU();
            if (!gpu)
                return nullptr;

            m_gpuForWebGPU = GPU::create(*gpu);
        }
    }

    return m_gpuForWebGPU.get();
#else
    return nullptr;
#endif
}

void WorkerNavigator::setAppBadge(std::optional<unsigned long long> badge, Ref<DeferredPromise>&& promise)
{
#if ENABLE(DECLARATIVE_WEB_PUSH)
    if (is<ServiceWorkerGlobalScope>(scriptExecutionContext())) {
        if (RefPtr declarativePushEvent = downcast<ServiceWorkerGlobalScope>(scriptExecutionContext())->declarativePushEvent()) {
            declarativePushEvent->setUpdatedAppBadge(WTFMove(badge));
            return;
        }
    }
#endif // ENABLE(DECLARATIVE_WEB_PUSH)

    auto* scope = downcast<WorkerGlobalScope>(scriptExecutionContext());
    if (!scope) {
        promise->reject(ExceptionCode::InvalidStateError);
        return;
    }

    if (auto* workerBadgeProxy = scope->thread().workerBadgeProxy())
        workerBadgeProxy->setAppBadge(badge);
    promise->resolve();
}

void WorkerNavigator::clearAppBadge(Ref<DeferredPromise>&& promise)
{
    setAppBadge(0, WTFMove(promise));
}

} // namespace WebCore
