/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 26, 2025.
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
#include "rtc_base/win/hstring.h"

#include <libloaderapi.h>
#include <winstring.h>

namespace {

FARPROC LoadComBaseFunction(const char* function_name) {
  static HMODULE const handle =
      ::LoadLibraryExW(L"combase.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
  return handle ? ::GetProcAddress(handle, function_name) : nullptr;
}

decltype(&::WindowsCreateString) GetWindowsCreateString() {
  static decltype(&::WindowsCreateString) const function =
      reinterpret_cast<decltype(&::WindowsCreateString)>(
          LoadComBaseFunction("WindowsCreateString"));
  return function;
}

decltype(&::WindowsDeleteString) GetWindowsDeleteString() {
  static decltype(&::WindowsDeleteString) const function =
      reinterpret_cast<decltype(&::WindowsDeleteString)>(
          LoadComBaseFunction("WindowsDeleteString"));
  return function;
}

}  // namespace

namespace webrtc {

bool ResolveCoreWinRTStringDelayload() {
  return GetWindowsDeleteString() && GetWindowsCreateString();
}

HRESULT CreateHstring(const wchar_t* src, uint32_t len, HSTRING* out_hstr) {
  decltype(&::WindowsCreateString) create_string_func =
      GetWindowsCreateString();
  if (!create_string_func)
    return E_FAIL;
  return create_string_func(src, len, out_hstr);
}

HRESULT DeleteHstring(HSTRING hstr) {
  decltype(&::WindowsDeleteString) delete_string_func =
      GetWindowsDeleteString();
  if (!delete_string_func)
    return E_FAIL;
  return delete_string_func(hstr);
}

}  // namespace webrtc
