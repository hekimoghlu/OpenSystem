/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 3, 2024.
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
#include "WorkerNotificationClient.h"

#if ENABLE(NOTIFICATIONS)

#include "NotificationData.h"
#include "NotificationResources.h"
#include "WorkerGlobalScope.h"
#include "WorkerLoaderProxy.h"
#include "WorkerThread.h"
#include <wtf/threads/BinarySemaphore.h>

namespace WebCore {

Ref<WorkerNotificationClient> WorkerNotificationClient::create(WorkerGlobalScope& workerScope)
{
    return adoptRef(*new WorkerNotificationClient(workerScope));
}

WorkerNotificationClient::WorkerNotificationClient(WorkerGlobalScope& workerScope)
    : m_workerScopeIdentifier(workerScope.identifier())
    , m_workerScope(workerScope)
{
}

bool WorkerNotificationClient::show(ScriptExecutionContext& workerContext, NotificationData&& notification, RefPtr<NotificationResources>&& resources, CompletionHandler<void()>&& completionHandler)
{
    auto callbackID = workerContext.addNotificationCallback(WTFMove(completionHandler));
    postToMainThread([protectedThis = Ref { *this }, notification = WTFMove(notification).isolatedCopy(), resources = WTFMove(resources), callbackID](auto* client, auto& context) mutable {
        if (!client) {
            protectedThis->postToWorkerThread([callbackID](auto& workerContext) {
                if (auto callback = workerContext.takeNotificationCallback(callbackID))
                    callback();
            });
            return;
        }
        client->show(context, WTFMove(notification), WTFMove(resources), [protectedThis = WTFMove(protectedThis), callbackID]() mutable {
            protectedThis->postToWorkerThread([callbackID](auto& workerContext) {
                if (auto callback = workerContext.takeNotificationCallback(callbackID))
                    callback();
            });
        });
    });
    return true;
}

void WorkerNotificationClient::cancel(NotificationData&& notification)
{
    postToMainThread([notification = WTFMove(notification).isolatedCopy()](auto* client, auto&) mutable {
        if (client)
            client->cancel(WTFMove(notification));
    });
}

void WorkerNotificationClient::notificationObjectDestroyed(NotificationData&& notification)
{
    postToMainThread([notification = WTFMove(notification).isolatedCopy()](auto* client, auto&) mutable {
        if (client)
            client->notificationObjectDestroyed(WTFMove(notification));
    });
}

void WorkerNotificationClient::notificationControllerDestroyed()
{
}

void WorkerNotificationClient::requestPermission(ScriptExecutionContext&, PermissionHandler&& completionHandler)
{
    // Workers cannot request permission at the moment.
    ASSERT_NOT_REACHED();
    completionHandler(Permission::Default);
}

auto WorkerNotificationClient::checkPermission(ScriptExecutionContext*) -> Permission
{
    Permission permission { Permission::Default };
    BinarySemaphore semaphore;
    postToMainThread([&permission, &semaphore](auto* client, auto& context) {
        if (client)
            permission = client->checkPermission(&context);
        semaphore.signal();
    });
    semaphore.wait();
    return permission;
}

void WorkerNotificationClient::postToMainThread(Function<void(NotificationClient*, ScriptExecutionContext& context)>&& task)
{
    m_workerScope.thread().workerLoaderProxy()->postTaskToLoader([task = WTFMove(task)](auto& context) mutable {
        task(context.notificationClient(), context);
    });
}

void WorkerNotificationClient::postToWorkerThread(Function<void(ScriptExecutionContext&)>&& task)
{
    ScriptExecutionContext::postTaskTo(m_workerScopeIdentifier, WTFMove(task));
}

} // namespace WebCore

#endif // ENABLE(NOTIFICATIONS)
