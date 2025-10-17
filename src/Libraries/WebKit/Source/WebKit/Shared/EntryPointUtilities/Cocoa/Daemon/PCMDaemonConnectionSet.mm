/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 2, 2025.
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
#import "PCMDaemonConnectionSet.h"

#import "PrivateClickMeasurementConnection.h"
#import "PrivateClickMeasurementManagerInterface.h"
#import <wtf/OSObjectPtr.h>
#import <wtf/RunLoop.h>

namespace WebKit::PCM {

DaemonConnectionSet& DaemonConnectionSet::singleton()
{
    ASSERT(RunLoop::isMain());
    static NeverDestroyed<DaemonConnectionSet> set;
    return set.get();
}
    
void DaemonConnectionSet::add(xpc_connection_t connection)
{
    ASSERT(!m_connections.contains(connection));
    m_connections.add(connection, DebugModeEnabled::No);
}

void DaemonConnectionSet::remove(xpc_connection_t connection)
{
    ASSERT(m_connections.contains(connection));
    auto hadDebugModeEnabled = m_connections.take(connection);
    if (hadDebugModeEnabled == DebugModeEnabled::Yes) {
        ASSERT(m_connectionsWithDebugModeEnabled);
        m_connectionsWithDebugModeEnabled--;
    }
}

void DaemonConnectionSet::setConnectedNetworkProcessHasDebugModeEnabled(const Daemon::Connection& connection, bool enabled)
{
    auto iterator = m_connections.find(connection.get());
    if (iterator == m_connections.end()) {
        ASSERT_NOT_REACHED();
        return;
    }
    bool wasEnabled = iterator->value == DebugModeEnabled::Yes;
    if (wasEnabled == enabled)
        return;

    if (enabled) {
        iterator->value = DebugModeEnabled::Yes;
        m_connectionsWithDebugModeEnabled++;
    } else {
        iterator->value = DebugModeEnabled::No;
        m_connectionsWithDebugModeEnabled--;
    }
}

bool DaemonConnectionSet::debugModeEnabled() const
{
    return !!m_connectionsWithDebugModeEnabled;
}

void DaemonConnectionSet::broadcastConsoleMessage(JSC::MessageLevel messageLevel, const String& message)
{
    auto dictionary = adoptOSObject(xpc_dictionary_create(nullptr, nullptr, 0));
    xpc_dictionary_set_uint64(dictionary.get(), protocolDebugMessageLevelKey, static_cast<uint64_t>(messageLevel));
    xpc_dictionary_set_string(dictionary.get(), protocolDebugMessageKey, message.utf8().data());
    for (auto& connection : m_connections.keys())
        xpc_connection_send_message(connection.get(), dictionary.get());
}

} // namespace WebKit::PCM
