/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 7, 2023.
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

#include "NotificationPermission.h"
#include <wtf/Forward.h>

namespace WebCore {

class NotificationPermissionCallback;
class NotificationResources;
class Page;
class ScriptExecutionContext;

struct NotificationData;

class NotificationClient {
public:
    using Permission = NotificationPermission;
    using PermissionHandler = CompletionHandler<void(Permission)>;

    // Requests that a notification be shown.
    virtual bool show(ScriptExecutionContext&, NotificationData&&, RefPtr<NotificationResources>&&, CompletionHandler<void()>&&) = 0;

    // Requests that a notification that has already been shown be canceled.
    virtual void cancel(NotificationData&&) = 0;

    // Informs the presenter that a Notification object has been destroyed
    // (such as by a page transition). The presenter may continue showing
    // the notification, but must not attempt to call the event handlers.
    virtual void notificationObjectDestroyed(NotificationData&&) = 0;

    // Informs the presenter the controller attached to the page has been destroyed.
    virtual void notificationControllerDestroyed() = 0;

    // Requests user permission to show desktop notifications from a particular
    // script context. The callback parameter should be run when the user has
    // made a decision.
    virtual void requestPermission(ScriptExecutionContext&, PermissionHandler&&) = 0;

    // Checks the current level of permission.
    virtual Permission checkPermission(ScriptExecutionContext*) = 0;

    virtual ~NotificationClient() = default;
};

WEBCORE_EXPORT void provideNotification(Page*, NotificationClient*);

} // namespace WebCore
