/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 26, 2024.
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
#import "LaunchServicesDatabaseManager.h"

#import "LaunchServicesDatabaseXPCConstants.h"
#import "Logging.h"
#import "XPCEndpoint.h"
#import <pal/spi/cocoa/LaunchServicesSPI.h>
#import <wtf/cocoa/Entitlements.h>
#import <wtf/spi/darwin/XPCSPI.h>
#import <wtf/text/WTFString.h>

namespace WebKit {

LaunchServicesDatabaseManager& LaunchServicesDatabaseManager::singleton()
{
    static LazyNeverDestroyed<LaunchServicesDatabaseManager> manager;
    static std::once_flag onceKey;
    std::call_once(onceKey, [] {
        manager.construct();
    });
    return manager.get();
}

void LaunchServicesDatabaseManager::handleEvent(xpc_object_t message)
{
    String messageName = xpc_dictionary_get_wtfstring(message, XPCEndpoint::xpcMessageNameKey);
    if (messageName == LaunchServicesDatabaseXPCConstants::xpcUpdateLaunchServicesDatabaseMessageName) {
#if HAVE(LSDATABASECONTEXT)
        auto database = xpc_dictionary_get_value(message, LaunchServicesDatabaseXPCConstants::xpcLaunchServicesDatabaseKey);

        RELEASE_LOG_FORWARDABLE(Loading, RECEIVED_LAUNCH_SERVICES_DATABASE);

        if (database)
            [LSDatabaseContext.sharedDatabaseContext observeDatabaseChange4WebKit:database];
#endif
        m_semaphore.signal();
        m_hasReceivedLaunchServicesDatabase = true;
    }
}

void LaunchServicesDatabaseManager::didConnect()
{
    auto message = adoptOSObject(xpc_dictionary_create(nullptr, nullptr, 0));
    xpc_dictionary_set_string(message.get(), XPCEndpoint::xpcMessageNameKey, LaunchServicesDatabaseXPCConstants::xpcRequestLaunchServicesDatabaseUpdateMessageName);

    auto connection = this->connection();
    if (!connection)
        return;

    xpc_connection_send_message(connection.get(), message.get());
}

bool LaunchServicesDatabaseManager::waitForDatabaseUpdate(Seconds timeout)
{
    if (m_hasReceivedLaunchServicesDatabase)
        return true;
    return m_semaphore.waitFor(timeout);
}

void LaunchServicesDatabaseManager::waitForDatabaseUpdate()
{
    auto startTime = MonotonicTime::now();
#ifdef NDEBUG
    constexpr auto waitTime = 5_s;
#else
    constexpr auto waitTime = 10_s;
#endif
    bool databaseUpdated = waitForDatabaseUpdate(waitTime);
    auto elapsedTime = MonotonicTime::now() - startTime;
    if (elapsedTime > 0.5_s)
        RELEASE_LOG_ERROR_FORWARDABLE(Loading, WAITING_FOR_LAUNCH_SERVICES_DATABASE_UPDATE_TOOK_F_SECONDS, elapsedTime.value());

    if (!databaseUpdated)
        RELEASE_LOG_FAULT_FORWARDABLE(Loading, TIMED_OUT_WAITING_FOR_LAUNCH_SERVICES_DATABASE_UPDATE);
}

}
