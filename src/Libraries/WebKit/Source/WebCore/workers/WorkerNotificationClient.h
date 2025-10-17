/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 4, 2021.
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

#if ENABLE(NOTIFICATIONS)

#include "NotificationClient.h"
#include "ScriptExecutionContextIdentifier.h"
#include <wtf/ThreadSafeRefCounted.h>

namespace WebCore {

class WorkerGlobalScope;

class WorkerNotificationClient : public NotificationClient, public ThreadSafeRefCounted<WorkerNotificationClient> {
public:
    static Ref<WorkerNotificationClient> create(WorkerGlobalScope&);

    // NotificationClient.
    bool show(ScriptExecutionContext&, NotificationData&&, RefPtr<NotificationResources>&&, CompletionHandler<void()>&&) final;
    void cancel(NotificationData&&) final;
    void notificationObjectDestroyed(NotificationData&&) final;
    void notificationControllerDestroyed() final;
    void requestPermission(ScriptExecutionContext&, PermissionHandler&&) final;
    Permission checkPermission(ScriptExecutionContext*) final;

private:
    explicit WorkerNotificationClient(WorkerGlobalScope&);

    void postToMainThread(Function<void(NotificationClient*, ScriptExecutionContext& context)>&&);
    void postToWorkerThread(Function<void(ScriptExecutionContext&)>&&);

    ScriptExecutionContextIdentifier m_workerScopeIdentifier;
    WorkerGlobalScope& m_workerScope;
};

} // namespace WebCore

#endif
