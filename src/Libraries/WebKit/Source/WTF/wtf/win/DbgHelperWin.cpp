/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 29, 2023.
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
#include <wtf/win/DbgHelperWin.h>

#include <mutex>

namespace WTF {

namespace DbgHelper {

// We are only calling these DbgHelp.dll functions in debug mode since the library is not threadsafe.
// It's possible for external code to call the library at the same time as WebKit and cause memory corruption.

#if !defined(NDEBUG)

static Lock callMutex;

static void initializeSymbols(HANDLE hProc)
{
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [&]() {
        if (!SymInitialize(hProc, nullptr, TRUE))
            LOG_ERROR("Failed to initialze symbol information %d", GetLastError());
    });
}

bool SymFromAddress(HANDLE hProc, DWORD64 address, DWORD64* displacement, SYMBOL_INFO* symbolInfo)
{
    Locker locker { callMutex };
    initializeSymbols(hProc);

    bool success = ::SymFromAddr(hProc, address, displacement, symbolInfo);
    if (success)
        symbolInfo->Name[symbolInfo->NameLen] = '\0';
    return success;
}

#else

bool SymFromAddress(HANDLE, DWORD64, DWORD64*, SYMBOL_INFO*)
{
    return false;
}

#endif // !defined(NDEBUG)

} // namespace DbgHelper

} // namespace WTF
