/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 8, 2021.
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
#include "SystemInfo.h"

#include <windows.h>
#include <wtf/text/MakeString.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

IGNORE_CLANG_WARNINGS_BEGIN("deprecated-declarations")

static std::tuple<int, int> windowsVersion()
{
    static bool initialized = false;
    static int majorVersion, minorVersion;

    if (!initialized) {
        initialized = true;
        OSVERSIONINFOEX versionInfo { };
        versionInfo.dwOSVersionInfoSize = sizeof(versionInfo);
        GetVersionEx(reinterpret_cast<OSVERSIONINFO*>(&versionInfo));
        majorVersion = versionInfo.dwMajorVersion;
        minorVersion = versionInfo.dwMinorVersion;
    }

    return { majorVersion, minorVersion };
}

IGNORE_CLANG_WARNINGS_END

static bool isWOW64()
{
    static bool initialized = false;
    static bool wow64 = false;

    if (!initialized) {
        initialized = true;
        HMODULE kernel32Module = GetModuleHandleA("kernel32.dll");
        if (!kernel32Module)
            return wow64;
        typedef BOOL (WINAPI* IsWow64ProcessFunc)(HANDLE, PBOOL);
        IsWow64ProcessFunc isWOW64Process = reinterpret_cast<IsWow64ProcessFunc>((void*)GetProcAddress(kernel32Module, "IsWow64Process"));
        if (isWOW64Process) {
            BOOL result = FALSE;
            wow64 = isWOW64Process(GetCurrentProcess(), &result) && result;
        }
    }

    return wow64;
}

static WORD processorArchitecture()
{
    static bool initialized = false;
    static WORD architecture = PROCESSOR_ARCHITECTURE_INTEL;

    if (!initialized) {
        initialized = true;
        SYSTEM_INFO systemInfo { };
        GetNativeSystemInfo(&systemInfo);
        architecture = systemInfo.wProcessorArchitecture;
    }
    return architecture;
}

static String architectureTokenForUAString()
{
    if (isWOW64())
        return "; WOW64"_s;
    if (processorArchitecture() == PROCESSOR_ARCHITECTURE_AMD64)
        return "; Win64; x64"_s;
    if (processorArchitecture() == PROCESSOR_ARCHITECTURE_IA64)
        return "; Win64; IA64"_s;
    return { };
}

String windowsVersionForUAString()
{
    auto [major, minor] = windowsVersion();
    return makeString("Windows NT "_s, major, '.', minor, architectureTokenForUAString());
}

} // namespace WebCore
