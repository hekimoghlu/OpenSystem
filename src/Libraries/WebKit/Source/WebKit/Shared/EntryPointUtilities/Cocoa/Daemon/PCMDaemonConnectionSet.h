/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 8, 2022.
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

#include <wtf/HashMap.h>
#include <wtf/RetainPtr.h>
#include <wtf/spi/darwin/XPCSPI.h>

namespace JSC {
enum class MessageLevel : uint8_t;
}

namespace WebKit {
namespace Daemon {
class Connection;
}
namespace PCM {

class DaemonConnectionSet {
public:
    static DaemonConnectionSet& singleton();
    
    void add(xpc_connection_t);
    void remove(xpc_connection_t);

    void setConnectedNetworkProcessHasDebugModeEnabled(const Daemon::Connection&, bool);
    bool debugModeEnabled() const;
    void broadcastConsoleMessage(JSC::MessageLevel, const String&);
    
private:
    enum class DebugModeEnabled : bool { No, Yes };
    HashMap<RetainPtr<xpc_connection_t>, DebugModeEnabled> m_connections;
    size_t m_connectionsWithDebugModeEnabled { 0 };
};

} // namespace PCM
} // namespace WebKit
