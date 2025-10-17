/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 17, 2023.
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
#import "config.h"
#import "WindowServerConnection.h"

#if PLATFORM(MAC) || PLATFORM(MACCATALYST)

#import "WebProcessPool.h"
#import <pal/spi/cg/CoreGraphicsSPI.h>

namespace WebKit {

#if PLATFORM(MAC)
void WindowServerConnection::applicationWindowModificationsStopped(bool stopped)
{
    if (m_applicationWindowModificationsHaveStopped == stopped)
        return;
    m_applicationWindowModificationsHaveStopped = stopped;
    windowServerConnectionStateChanged();
}

void WindowServerConnection::windowServerConnectionStateChanged()
{
    for (auto& processPool : WebProcessPool::allProcessPools())
        processPool->windowServerConnectionStateChanged();
}
#endif

void WindowServerConnection::hardwareConsoleStateChanged(HardwareConsoleState state)
{
    if (state == m_connectionState)
        return;

    m_connectionState = state;
    for (auto& processPool : WebProcessPool::allProcessPools())
        processPool->hardwareConsoleStateChanged();
}

WindowServerConnection& WindowServerConnection::singleton()
{
    static WindowServerConnection& windowServerConnection = *new WindowServerConnection;
    return windowServerConnection;
}

#if PLATFORM(MAC)
static bool registerOcclusionNotificationHandler(CGSNotificationType type, CGSNotifyConnectionProcPtr handler)
{
    CGSConnectionID mainConnection = CGSMainConnectionID();
    static bool notificationsEnabled;
    if (!notificationsEnabled) {
        if (CGSPackagesEnableConnectionOcclusionNotifications(mainConnection, true, nullptr) != kCGErrorSuccess)
            return false;
        if (CGSPackagesEnableConnectionWindowModificationNotifications(mainConnection, true, nullptr) != kCGErrorSuccess)
            return false;
        notificationsEnabled = true;
    }

    return CGSRegisterConnectionNotifyProc(mainConnection, handler, type, nullptr) == kCGErrorSuccess;
}
#endif

WindowServerConnection::WindowServerConnection()
    : m_applicationWindowModificationsHaveStopped(false)
{
#if PLATFORM(MAC)
    struct OcclusionNotificationHandler {
        CGSNotificationType notificationType;
        CGSNotifyConnectionProcPtr handler;
        ASCIILiteral name;
    };

    static auto windowModificationsStarted = [](CGSNotificationType, void*, uint32_t, void*, CGSConnectionID) {
        WindowServerConnection::singleton().applicationWindowModificationsStopped(false);
    };

    static auto windowModificationsStopped = [](CGSNotificationType, void*, uint32_t, void*, CGSConnectionID) {
        WindowServerConnection::singleton().applicationWindowModificationsStopped(true);
    };

    static const OcclusionNotificationHandler occlusionNotificationHandlers[] = {
        { kCGSConnectionWindowModificationsStarted, windowModificationsStarted, "Application Window Modifications Started"_s },
        { kCGSConnectionWindowModificationsStopped, windowModificationsStopped, "Application Window Modifications Stopped"_s },
    };

    for (const auto& occlusionNotificationHandler : occlusionNotificationHandlers) {
        bool result = registerOcclusionNotificationHandler(occlusionNotificationHandler.notificationType, occlusionNotificationHandler.handler);
        UNUSED_PARAM(result);
        ASSERT_WITH_MESSAGE(result, "Registration of \"%s\" notification handler failed.\n", occlusionNotificationHandler.name.characters());
    }
#endif

    static auto consoleWillDisconnect = [](CGSNotificationType, void*, uint32_t, void*) {
        WindowServerConnection::singleton().hardwareConsoleStateChanged(HardwareConsoleState::Disconnected);
    };
    static auto consoleWillConnect = [](CGSNotificationType, void*, uint32_t, void*) {
        WindowServerConnection::singleton().hardwareConsoleStateChanged(HardwareConsoleState::Connected);
    };

    struct ConnectionStateNotificationHandler {
        CGSNotificationType notificationType;
        CGSNotifyProcPtr handler;
        ASCIILiteral name;
    };

    static const ConnectionStateNotificationHandler connectionStateNotificationHandlers[] = {
        { kCGSessionConsoleWillDisconnect, consoleWillDisconnect, "Console Disconnected"_s },
        { kCGSessionConsoleConnect, consoleWillConnect, "Console Connected"_s },
    };

    for (const auto& connectionStateNotificationHandler : connectionStateNotificationHandlers) {
        auto error = CGSRegisterNotifyProc(connectionStateNotificationHandler.handler, connectionStateNotificationHandler.notificationType, nullptr);
        UNUSED_PARAM(error);
        ASSERT_WITH_MESSAGE(!error, "Registration of \"%s\" notification handler failed.\n", connectionStateNotificationHandler.name.characters());
    }
}

} // namespace WebKit

#endif // PLATFORM(MAC) || PLATFORM(MACCATALYST)
