/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 6, 2022.
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

#include "Connection.h"
#include "IPCUtilities.h"
#include <shlwapi.h>
#include <wtf/RunLoop.h>
#include <wtf/text/StringBuilder.h>

namespace WebKit {

static LPCWSTR processName(ProcessLauncher::ProcessType processType)
{
    switch (processType) {
    case ProcessLauncher::ProcessType::Web:
        return L"WebKitWebProcess.exe";
    case ProcessLauncher::ProcessType::Network:
        return L"WebKitNetworkProcess.exe";
#if ENABLE(GPU_PROCESS)
    case ProcessLauncher::ProcessType::GPU:
        return L"WebKitGPUProcess.exe";
#endif
    }
    return L"WebKitWebProcess.exe";
}

void ProcessLauncher::launchProcess()
{
    // First, create the server and client identifiers.
    HANDLE serverIdentifier, clientIdentifier;
    if (!IPC::createServerAndClientIdentifiers(serverIdentifier, clientIdentifier)) {
        // FIXME: What should we do here?
        ASSERT_NOT_REACHED();
    }

    // Ensure that the child process inherits the client identifier.
    ::SetHandleInformation(clientIdentifier, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT);

    // To get the full file path to WebKit2WebProcess.exe, we fild the location of WebKit2.dll,
    // remove the last path component.
    HMODULE webKitModule = ::GetModuleHandle(L"WebKit2.dll");
    ASSERT(webKitModule);
    if (!webKitModule)
        return;

    WCHAR pathStr[MAX_PATH];
    if (!::GetModuleFileName(webKitModule, pathStr, std::size(pathStr)))
        return;

    ::PathRemoveFileSpec(pathStr);
    if (!::PathAppend(pathStr, processName(m_launchOptions.processType)))
        return;

    StringBuilder commandLineBuilder;
    commandLineBuilder.append("\""_s);
    commandLineBuilder.append(String(pathStr));
    commandLineBuilder.append("\""_s);
    commandLineBuilder.append(" -type "_s);
    commandLineBuilder.append(String::number(static_cast<int>(m_launchOptions.processType)));
    commandLineBuilder.append(" -processIdentifier "_s);
    commandLineBuilder.append(String::number(m_launchOptions.processIdentifier.toUInt64()));
    commandLineBuilder.append(" -clientIdentifier "_s);
    commandLineBuilder.append(String::number(reinterpret_cast<uintptr_t>(clientIdentifier)));
    if (m_client->shouldConfigureJSCForTesting())
        commandLineBuilder.append(" -configure-jsc-for-testing"_s);
    if (!m_client->isJITEnabled())
        commandLineBuilder.append(" -disable-jit"_s);
    commandLineBuilder.append('\0');

    auto commandLine = commandLineBuilder.toString().wideCharacters();

    STARTUPINFO startupInfo { };
    startupInfo.cb = sizeof(startupInfo);
    startupInfo.dwFlags = STARTF_USESHOWWINDOW;
    startupInfo.wShowWindow = SW_HIDE;
    PROCESS_INFORMATION processInformation { };
    BOOL result = ::CreateProcess(0, commandLine.data(), 0, 0, true, 0, 0, 0, &startupInfo, &processInformation);

    // We can now close the client identifier handle.
    ::CloseHandle(clientIdentifier);

    if (!result) {
        // FIXME: What should we do here?
        ASSERT_NOT_REACHED();
    }

    // Don't leak the thread handle.
    ::CloseHandle(processInformation.hThread);

    // We've finished launching the process, message back to the run loop.
    RefPtr<ProcessLauncher> protectedThis(this);
    m_hProcess = Win32Handle::adopt(processInformation.hProcess);
    WTF::ProcessID pid = processInformation.dwProcessId;

    RunLoop::main().dispatch([protectedThis, pid, serverIdentifier] {
        protectedThis->didFinishLaunchingProcess(pid, IPC::Connection::Identifier { serverIdentifier });
    });
}

void ProcessLauncher::terminateProcess()
{
    if (!m_hProcess)
        return;

    ::TerminateProcess(m_hProcess.get(), 0);
}

void ProcessLauncher::platformInvalidate()
{
    m_hProcess = { };
}

} // namespace WebKit
