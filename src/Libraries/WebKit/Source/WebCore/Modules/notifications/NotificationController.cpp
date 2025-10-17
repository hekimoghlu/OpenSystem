/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 12, 2023.
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
#include "NotificationController.h"

#if ENABLE(NOTIFICATIONS)

#include "NotificationClient.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(NotificationController);

NotificationController::NotificationController(NotificationClient* client)
    : m_client(*client)
{
    ASSERT(client);
}

NotificationController::~NotificationController()
{
    m_client.notificationControllerDestroyed();
}

NotificationClient* NotificationController::clientFrom(Page& page)
{
    auto* controller = NotificationController::from(&page);
    if (!controller)
        return nullptr;
    return &controller->client();
}

ASCIILiteral NotificationController::supplementName()
{
    return "NotificationController"_s;
}

void provideNotification(Page* page, NotificationClient* client)
{
    NotificationController::provideTo(page, NotificationController::supplementName(), makeUnique<NotificationController>(client));
}

} // namespace WebCore

#endif // ENABLE(NOTIFICATIONS)
