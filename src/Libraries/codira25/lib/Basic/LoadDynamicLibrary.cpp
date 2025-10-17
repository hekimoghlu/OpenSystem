/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 6, 2023.
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

//===----------------------------------------------------------------------===//
//
// Copyright (c) NeXTHub Corporation. All rights reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
//
// This code is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// version 2 for more details (a copy is included in the LICENSE file that
// accompanied this code).
//
// Author(-s): Tunjay Akbarli
//

//===----------------------------------------------------------------------===//

#include "language/Basic/LoadDynamicLibrary.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include "toolchain/Support/ConvertUTF.h"
#include "toolchain/Support/Windows/WindowsSupport.h"
#include "language/Basic/Toolchain.h"
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#if defined(_WIN32)
void *language::loadLibrary(const char *path, std::string *err) {
  SmallVector<wchar_t, MAX_PATH> pathUnicode;
  if (std::error_code ec = toolchain::sys::windows::UTF8ToUTF16(path, pathUnicode)) {
    SetLastError(ec.value());
    toolchain::MakeErrMsg(err, std::string(path) + ": Can't convert to UTF-16");
    return nullptr;
  }

  HMODULE handle = LoadLibraryW(pathUnicode.data());
  if (handle == NULL) {
    toolchain::MakeErrMsg(err, std::string(path) + ": Can't open");
    return nullptr;
  }
  return (void *)handle;
}

void *language::getAddressOfSymbol(void *handle, const char *symbol) {
  return (void *)uintptr_t(GetProcAddress((HMODULE)handle, symbol));
}

#else
void *language::loadLibrary(const char *path, std::string *err) {
  void *handle = ::dlopen(path, RTLD_LAZY | RTLD_LOCAL);
  if (!handle)
    *err = ::dlerror();
  return handle;
}

void *language::getAddressOfSymbol(void *handle, const char *symbol) {
  return ::dlsym(handle, symbol);
}
#endif