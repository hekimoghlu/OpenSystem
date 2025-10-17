/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 11, 2022.
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
#include "ProcessLauncher.h"

#include <wtf/StdLibExtras.h>
#include <wtf/SystemTracing.h>
#include <wtf/WorkQueue.h>

#if OS(DARWIN)
#include <mach/mach_init.h>
#include <mach/mach_traps.h>
#endif

namespace WebKit {

ProcessLauncher::ProcessLauncher(Client* client, LaunchOptions&& launchOptions)
    : m_client(client)
    , m_launchOptions(WTFMove(launchOptions))
{
    tracePoint(ProcessLaunchStart, m_launchOptions.processIdentifier.toUInt64());
    launchProcess();
}

ProcessLauncher::~ProcessLauncher()
{
    platformDestroy();

    if (m_isLaunching)
        tracePoint(ProcessLaunchEnd, m_launchOptions.processIdentifier.toUInt64(), static_cast<uint64_t>(m_launchOptions.processType));
}

#if !PLATFORM(COCOA)
void ProcessLauncher::platformDestroy()
{
}
#endif

void ProcessLauncher::didFinishLaunchingProcess(ProcessID processIdentifier, IPC::Connection::Identifier&& identifier)
{
    m_processID = processIdentifier;
    m_isLaunching = false;

    tracePoint(ProcessLaunchEnd, m_launchOptions.processIdentifier.toUInt64(), static_cast<uint64_t>(m_launchOptions.processType), static_cast<uint64_t>(m_processID));

    if (!m_client) {
#if OS(DARWIN) && !USE(UNIX_DOMAIN_SOCKETS)
        // FIXME: Release port rights/connections in the Connection::Identifier destructor.
        if (identifier.port)
            mach_port_mod_refs(mach_task_self(), identifier.port, MACH_PORT_RIGHT_RECEIVE, -1);
#endif
        return;
    }
    
    m_client->didFinishLaunching(this, WTFMove(identifier));
}

void ProcessLauncher::invalidate()
{
    m_client = nullptr;
    platformInvalidate();
}

} // namespace WebKit
