/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 17, 2024.
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

#include <windows.h>
#include <wtf/FastMalloc.h>

namespace WTF {

class Win32Handle {
    WTF_MAKE_FAST_ALLOCATED;
public:
    WTF_EXPORT_PRIVATE static Win32Handle adopt(HANDLE);

    Win32Handle() = default;
    WTF_EXPORT_PRIVATE explicit Win32Handle(const Win32Handle&);
    WTF_EXPORT_PRIVATE Win32Handle(Win32Handle&&);
    WTF_EXPORT_PRIVATE ~Win32Handle();

    WTF_EXPORT_PRIVATE Win32Handle& operator=(Win32Handle&&);

    explicit operator bool() const { return m_handle != INVALID_HANDLE_VALUE; }

    HANDLE get() const { return m_handle; }

    WTF_EXPORT_PRIVATE HANDLE leak() WARN_UNUSED_RETURN;

    struct IPCData {
        uintptr_t handle;
        DWORD processID;
    };

    WTF_EXPORT_PRIVATE IPCData toIPCData();
    WTF_EXPORT_PRIVATE static Win32Handle createFromIPCData(IPCData&&);

private:
    explicit Win32Handle(HANDLE);

    HANDLE m_handle { INVALID_HANDLE_VALUE };
};

} // namespace WTF

using WTF::Win32Handle;
