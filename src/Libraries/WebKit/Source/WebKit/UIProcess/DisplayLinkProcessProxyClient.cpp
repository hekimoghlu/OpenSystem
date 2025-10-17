/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 18, 2021.
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
#include "DisplayLinkProcessProxyClient.h"

#if HAVE(DISPLAY_LINK)

#include "EventDispatcherMessages.h"
#include "WebProcessMessages.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(DisplayLinkProcessProxyClient);

void DisplayLinkProcessProxyClient::setConnection(RefPtr<IPC::Connection>&& connection)
{
    Locker locker { m_connectionLock };
    m_connection = WTFMove(connection);
}

// This is called off the main thread.
void DisplayLinkProcessProxyClient::displayLinkFired(WebCore::PlatformDisplayID displayID, WebCore::DisplayUpdate displayUpdate, bool wantsFullSpeedUpdates, bool anyObserverWantsCallback)
{
    RefPtr<IPC::Connection> connection;
    {
        Locker locker { m_connectionLock };
        connection = m_connection;
    }

    if (!connection)
        return;

    if (wantsFullSpeedUpdates)
        connection->send(Messages::EventDispatcher::DisplayDidRefresh(displayID, displayUpdate, anyObserverWantsCallback), 0, { }, Thread::QOS::UserInteractive);
    else if (anyObserverWantsCallback)
        connection->send(Messages::WebProcess::DisplayDidRefresh(displayID, displayUpdate), 0, { }, Thread::QOS::UserInteractive);
}

} // namespace WebKit

#endif // HAVE(DISPLAY_LINK)
