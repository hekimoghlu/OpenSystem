/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 9, 2024.
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
#ifndef WindowsExtras_h
#define WindowsExtras_h

#if OS(WINDOWS)

#include <windows.h>
#include <objbase.h>
#include <shlwapi.h>

#ifndef HWND_MESSAGE
const HWND HWND_MESSAGE = 0;
#endif

namespace WTF {

inline HRESULT getRegistryValue(HKEY hkey, LPCWSTR pszSubKey, LPCWSTR pszValue, LPDWORD pdwType, LPVOID pvData, LPDWORD pcbData)
{
    return ::SHGetValueW(hkey, pszSubKey, pszValue, pdwType, pvData, pcbData);
}

inline void* getWindowPointer(HWND hWnd, int index)
{
    return reinterpret_cast<void*>(::GetWindowLongPtr(hWnd, index));
}

inline void* setWindowPointer(HWND hWnd, int index, void* value)
{
    return reinterpret_cast<void*>(::SetWindowLongPtr(hWnd, index, reinterpret_cast<LONG_PTR>(value)));
}

} // namespace WTF

using WTF::getRegistryValue;
using WTF::getWindowPointer;
using WTF::setWindowPointer;

#endif // OS(WINDOWS)

#endif // WindowsExtras_h
