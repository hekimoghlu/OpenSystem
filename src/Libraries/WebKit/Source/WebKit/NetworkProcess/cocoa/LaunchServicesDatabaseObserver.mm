/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 14, 2024.
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
#import "LaunchServicesDatabaseObserver.h"

#import "LaunchServicesDatabaseXPCConstants.h"
#import <pal/spi/cocoa/LaunchServicesSPI.h>
#import <wtf/BlockPtr.h>
#import <wtf/TZoneMallocInlines.h>
#import <wtf/cocoa/Entitlements.h>
#import <wtf/spi/cocoa/SecuritySPI.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(LaunchServicesDatabaseObserver);

LaunchServicesDatabaseObserver::LaunchServicesDatabaseObserver(NetworkProcess&)
{
#if HAVE(LSDATABASECONTEXT) && !HAVE(SYSTEM_CONTENT_LS_DATABASE)
    m_observer = [LSDatabaseContext.sharedDatabaseContext addDatabaseChangeObserver4WebKit:^(xpc_object_t change) {
        auto message = adoptOSObject(xpc_dictionary_create(nullptr, nullptr, 0));
        xpc_dictionary_set_string(message.get(), XPCEndpoint::xpcMessageNameKey, LaunchServicesDatabaseXPCConstants::xpcUpdateLaunchServicesDatabaseMessageName);
        xpc_dictionary_set_value(message.get(), LaunchServicesDatabaseXPCConstants::xpcLaunchServicesDatabaseKey, change);

        Locker locker { m_connectionsLock };
        for (auto& connection : m_connections) {
            RELEASE_ASSERT(xpc_get_type(connection.get()) == XPC_TYPE_CONNECTION);
            xpc_connection_send_message(connection.get(), message.get());
        }
    }];
#endif
}

ASCIILiteral LaunchServicesDatabaseObserver::supplementName()
{
    return "LaunchServicesDatabaseObserverSupplement"_s;
}

void LaunchServicesDatabaseObserver::startObserving(OSObjectPtr<xpc_connection_t> connection)
{
    {
        Locker locker { m_connectionsLock };
        m_connections.append(connection);
    }

#if HAVE(SYSTEM_CONTENT_LS_DATABASE)
    [LSDatabaseContext.sharedDatabaseContext getSystemContentDatabaseObject4WebKit:makeBlockPtr([connection = connection] (xpc_object_t _Nullable object, NSError * _Nullable error) {
        if (!object)
            return;
        auto message = adoptOSObject(xpc_dictionary_create(nullptr, nullptr, 0));
        xpc_dictionary_set_string(message.get(), XPCEndpoint::xpcMessageNameKey, LaunchServicesDatabaseXPCConstants::xpcUpdateLaunchServicesDatabaseMessageName);
        xpc_dictionary_set_value(message.get(), LaunchServicesDatabaseXPCConstants::xpcLaunchServicesDatabaseKey, object);

        xpc_connection_send_message(connection.get(), message.get());

    }).get()];
#elif HAVE(LSDATABASECONTEXT)
    RetainPtr<id> observer = [LSDatabaseContext.sharedDatabaseContext addDatabaseChangeObserver4WebKit:^(xpc_object_t change) {
        auto message = adoptOSObject(xpc_dictionary_create(nullptr, nullptr, 0));
        xpc_dictionary_set_string(message.get(), XPCEndpoint::xpcMessageNameKey, LaunchServicesDatabaseXPCConstants::xpcUpdateLaunchServicesDatabaseMessageName);
        xpc_dictionary_set_value(message.get(), LaunchServicesDatabaseXPCConstants::xpcLaunchServicesDatabaseKey, change);

        xpc_connection_send_message(connection.get(), message.get());
    }];

    [LSDatabaseContext.sharedDatabaseContext removeDatabaseChangeObserver4WebKit:observer.get()];
#else
    auto message = adoptOSObject(xpc_dictionary_create(nullptr, nullptr, 0));
    xpc_dictionary_set_string(message.get(), XPCEndpoint::xpcMessageNameKey, LaunchServicesDatabaseXPCConstants::xpcUpdateLaunchServicesDatabaseMessageName);
    xpc_connection_send_message(connection.get(), message.get());
#endif
}

LaunchServicesDatabaseObserver::~LaunchServicesDatabaseObserver()
{
#if HAVE(LSDATABASECONTEXT) && !HAVE(SYSTEM_CONTENT_LS_DATABASE)
    [LSDatabaseContext.sharedDatabaseContext removeDatabaseChangeObserver4WebKit:m_observer.get()];
#endif
}

ASCIILiteral LaunchServicesDatabaseObserver::xpcEndpointMessageNameKey() const
{
    return xpcMessageNameKey;
}

ASCIILiteral LaunchServicesDatabaseObserver::xpcEndpointMessageName() const
{
    return LaunchServicesDatabaseXPCConstants::xpcLaunchServicesDatabaseXPCEndpointMessageName;
}

ASCIILiteral LaunchServicesDatabaseObserver::xpcEndpointNameKey() const
{
    return LaunchServicesDatabaseXPCConstants::xpcLaunchServicesDatabaseXPCEndpointNameKey;
}

void LaunchServicesDatabaseObserver::handleEvent(xpc_connection_t connection, xpc_object_t event)
{
    if (xpc_get_type(event) == XPC_TYPE_ERROR) {
        if (event != XPC_ERROR_CONNECTION_INVALID && event != XPC_ERROR_TERMINATION_IMMINENT)
            return;

        Locker locker { m_connectionsLock };
        for (size_t i = 0; i < m_connections.size(); i++) {
            if (m_connections[i].get() == connection) {
                m_connections.remove(i);
                break;
            }
        }
        return;
    }
    if (xpc_get_type(event) == XPC_TYPE_DICTIONARY) {
        String messageName = xpc_dictionary_get_wtfstring(event, xpcMessageNameKey);
        if (messageName != LaunchServicesDatabaseXPCConstants::xpcRequestLaunchServicesDatabaseUpdateMessageName)
            return;
        startObserving(connection);
    }
}

void LaunchServicesDatabaseObserver::initializeConnection(IPC::Connection* connection)
{
    sendEndpointToConnection(connection->xpcConnection());
}

}
