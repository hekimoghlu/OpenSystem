/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 4, 2025.
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

#include "IPCUtilities.h"
#include <stdint.h>
#include <sys/socket.h>

#if USE(WPE_BACKEND_PLAYSTATION)
#include "ProcessProviderLibWPE.h"
#else
#include <process-launcher.h>
#endif

namespace WebKit {

#if !USE(WPE_BACKEND_PLAYSTATION)
#define MAKE_PROCESS_PATH(x) "/app0/" #x "Process.self"
static const char* defaultProcessPath(ProcessLauncher::ProcessType processType)
{
    switch (processType) {
    case ProcessLauncher::ProcessType::Network:
        return MAKE_PROCESS_PATH(Network);
#if ENABLE(GPU_PROCESS)
    case ProcessLauncher::ProcessType::GPU:
        return MAKE_PROCESS_PATH(GPU);
#endif
    case ProcessLauncher::ProcessType::Web:
    default:
        return MAKE_PROCESS_PATH(Web);
    }
}
#endif

void ProcessLauncher::launchProcess()
{
    IPC::SocketPair socketPair = IPC::createPlatformConnection(IPC::PlatformConnectionOptions::SetCloexecOnServer);

    int sendBufSize = 32 * 1024;
    setsockopt(socketPair.server.value(), SOL_SOCKET, SO_SNDBUF, &sendBufSize, 4);
    setsockopt(socketPair.client.value(), SOL_SOCKET, SO_SNDBUF, &sendBufSize, 4);

    int recvBufSize = 32 * 1024;
    setsockopt(socketPair.server.value(), SOL_SOCKET, SO_RCVBUF, &recvBufSize, 4);
    setsockopt(socketPair.client.value(), SOL_SOCKET, SO_RCVBUF, &recvBufSize, 4);

    char coreProcessIdentifierString[16];
    snprintf(coreProcessIdentifierString, sizeof coreProcessIdentifierString, "%ld", m_launchOptions.processIdentifier.toUInt64());

    char* argv[] = {
        coreProcessIdentifierString,
        nullptr
    };

#if USE(WPE_BACKEND_PLAYSTATION)
    auto appLocalPid = ProcessProviderLibWPE::singleton().launchProcess(m_launchOptions, argv, socketPair.client.value());
#else
    PlayStation::LaunchParam param { socketPair.client.value(), m_launchOptions.userId };
    int32_t appLocalPid = PlayStation::launchProcess(
        !m_launchOptions.processPath.isEmpty() ? m_launchOptions.processPath.utf8().data() : defaultProcessPath(m_launchOptions.processType),
        argv, param);
#endif

    if (appLocalPid < 0) {
#ifndef NDEBUG
        fprintf(stderr, "Failed to launch process. err=0x%08x path=%s\n", appLocalPid, m_launchOptions.processPath.utf8().data());
#endif
        return;
    }

    // We've finished launching the process, message back to the main run loop.
    RunLoop::main().dispatch([protectedThis = Ref { *this }, appLocalPid, serverIdentifier = WTFMove(socketPair.server)] mutable {
        protectedThis->didFinishLaunchingProcess(appLocalPid, IPC::Connection::Identifier { WTFMove(serverIdentifier) });
    });
}

void ProcessLauncher::terminateProcess()
{
    if (!m_processID)
        return;

#if USE(WPE_BACKEND_PLAYSTATION)
    ProcessProviderLibWPE::singleton().kill(m_processID);
#else
    PlayStation::terminateProcess(m_processID);
#endif
}

void ProcessLauncher::platformInvalidate()
{
    m_processID = 0;
}

} // namespace WebKit
