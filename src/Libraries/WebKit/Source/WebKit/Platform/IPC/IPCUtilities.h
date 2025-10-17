/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 15, 2022.
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

#define ASSERT_IS_TESTING_IPC() ASSERT(IPC::isTestingIPC(), "Untrusted connection sent invalid data. Should only happen when testing IPC.")

#if OS(WINDOWS)
#include <windows.h>
#endif

#if USE(UNIX_DOMAIN_SOCKETS)
#include <wtf/unix/UnixFileDescriptor.h>
#endif

namespace IPC {

// Function to check when asserting IPC-related failures, so that IPC testing skips the assertions
// and exposes bugs underneath.
bool isTestingIPC();

#if ENABLE(IPC_TESTING_API)
void startTestingIPC();
void stopTestingIPC();
#else
inline bool isTestingIPC()
{
    return false;
}
#endif

#if USE(UNIX_DOMAIN_SOCKETS)
struct SocketPair {
    UnixFileDescriptor client;
    UnixFileDescriptor server;
};

enum PlatformConnectionOptions {
    SetCloexecOnClient = 1 << 0,
    SetCloexecOnServer = 1 << 1,
#if USE(GLIB) && OS(LINUX)
    SetPasscredOnServer = 1 << 2
#endif
};

SocketPair createPlatformConnection(unsigned options = SetCloexecOnClient | SetCloexecOnServer);

#if USE(GLIB) && OS(LINUX)
void sendPIDToPeer(int socket);
pid_t readPIDFromPeer(int socket);
#endif
#endif

#if OS(WINDOWS)
bool createServerAndClientIdentifiers(HANDLE& serverIdentifier, HANDLE& clientIdentifier);
#endif

}
