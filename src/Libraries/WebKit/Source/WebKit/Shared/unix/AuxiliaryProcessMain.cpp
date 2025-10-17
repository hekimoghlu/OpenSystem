/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 24, 2021.
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
#include "AuxiliaryProcessMain.h"

#include "IPCUtilities.h"
#include <JavaScriptCore/Options.h>
#include <WebCore/ProcessIdentifier.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <wtf/text/StringToIntegerConversion.h>

#if ENABLE(BREAKPAD)
#include "unix/BreakpadExceptionHandler.h"
#endif

namespace WebKit {

AuxiliaryProcessMainCommon::AuxiliaryProcessMainCommon()
{
#if ENABLE(BREAKPAD)
    installBreakpadExceptionHandler();
#endif
}

// The command line is constructed in ProcessLauncher::launchProcess.
WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN // Unix port
bool AuxiliaryProcessMainCommon::parseCommandLine(int argc, char** argv)
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
{
    int argIndex = 1; // Start from argv[1], since argv[0] is the program name.

    // Ensure we have enough arguments for processIdentifier and connectionIdentifier
    if (argc < argIndex + 2)
        return false;

    if (auto processIdentifier = parseInteger<uint64_t>(unsafeSpan(argv[argIndex++]))) {
        if (!ObjectIdentifier<WebCore::ProcessIdentifierType>::isValidIdentifier(*processIdentifier))
            return false;
        m_parameters.processIdentifier = ObjectIdentifier<WebCore::ProcessIdentifierType>(*processIdentifier);
    } else
        return false;

    if (auto connectionIdentifier = parseInteger<int>(unsafeSpan(argv[argIndex++])))
        m_parameters.connectionIdentifier = IPC::Connection::Identifier { { *connectionIdentifier, UnixFileDescriptor::Adopt } };
    else
        return false;

    if (!m_parameters.processIdentifier->toRawValue() || m_parameters.connectionIdentifier.handle.value() <= 0)
        return false;

#if USE(GLIB) && OS(LINUX)
    // Parse pidSocket if available
    if (argc > argIndex) {
        auto pidSocket = parseInteger<int>(unsafeSpan(argv[argIndex]));
        if (pidSocket && *pidSocket >= 0) {
            IPC::sendPIDToPeer(*pidSocket);
            RELEASE_ASSERT(!close(*pidSocket));
            ++argIndex;
        } else
            return false;
    }
#endif

#if ENABLE(DEVELOPER_MODE)
    // Check last remaining options for JSC testing
    for (; argIndex < argc; ++argIndex) {
        if (argv[argIndex] && !strcmp(argv[argIndex], "--configure-jsc-for-testing"))
            JSC::Config::configureForTesting();
    }
#endif

    return true;
}

void AuxiliaryProcess::platformInitialize(const AuxiliaryProcessInitializationParameters&)
{
    struct sigaction signalAction;
    memset(&signalAction, 0, sizeof(signalAction));
    RELEASE_ASSERT(!sigemptyset(&signalAction.sa_mask));
    signalAction.sa_handler = SIG_IGN;
    RELEASE_ASSERT(!sigaction(SIGPIPE, &signalAction, nullptr));
}

} // namespace WebKit
