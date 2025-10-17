/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 7, 2024.
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
#include <wtf/win/Win32Handle.h>

namespace WTF {

static void closeHandle(HANDLE handle)
{
    if (handle == INVALID_HANDLE_VALUE)
        return;

    ::CloseHandle(handle);
}

static HANDLE duplicateHandle(HANDLE handle)
{
    if (handle == INVALID_HANDLE_VALUE)
        return INVALID_HANDLE_VALUE;

    auto processHandle = ::GetCurrentProcess();
    HANDLE duplicate;
    if (!::DuplicateHandle(processHandle, handle, processHandle, &duplicate, 0, FALSE, DUPLICATE_SAME_ACCESS))
        return INVALID_HANDLE_VALUE;

    return duplicate;
}

Win32Handle Win32Handle::adopt(HANDLE handle)
{
    return Win32Handle(handle);
}

Win32Handle::Win32Handle(HANDLE handle)
    : m_handle(handle)
{
}

Win32Handle::Win32Handle(const Win32Handle& other)
    : m_handle(duplicateHandle(other.get()))
{
}

Win32Handle::Win32Handle(Win32Handle&& other)
    : m_handle(other.leak())
{
}

Win32Handle::~Win32Handle()
{
    closeHandle(m_handle);
}

Win32Handle& Win32Handle::operator=(Win32Handle&& other)
{
    if (this != &other) {
        closeHandle(m_handle);
        m_handle = other.leak();
    }

    return *this;
}

HANDLE Win32Handle::leak()
{
    return std::exchange(m_handle, INVALID_HANDLE_VALUE);
}

Win32Handle::IPCData Win32Handle::toIPCData()
{
    return { reinterpret_cast<uintptr_t>(leak()), GetCurrentProcessId() };
}

Win32Handle Win32Handle::createFromIPCData(IPCData&& data)
{
    auto handle = reinterpret_cast<HANDLE>(data.handle);
    if (handle == INVALID_HANDLE_VALUE)
        return { };
    auto sourceProcess = Win32Handle::adopt(OpenProcess(PROCESS_DUP_HANDLE, FALSE, data.processID));
    if (!sourceProcess)
        return { };
    HANDLE duplicatedHandle;
    // Copy the handle into our process and close the handle that the sending process created for us.
    if (!DuplicateHandle(sourceProcess.get(), handle, GetCurrentProcess(), &duplicatedHandle, 0, FALSE, DUPLICATE_SAME_ACCESS | DUPLICATE_CLOSE_SOURCE)) {
        // The source process may exit after the above OpenProcess calling.
        // DuplicateHandle fails with ERROR_INVALID_HANDLE in such a case.
        return { };
    }
    return Win32Handle::adopt(duplicatedHandle);
}

} // namespace WTF
