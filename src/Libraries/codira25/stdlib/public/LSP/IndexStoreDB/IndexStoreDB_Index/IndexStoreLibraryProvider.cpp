/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 9, 2023.
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

//===--- IndexStoreLibraryProvider.cpp ------------------------------------===//
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

#include <IndexStoreDB_Index/IndexStoreLibraryProvider.h>
#include <IndexStoreDB_Index/IndexStoreCXX.h>
#include <IndexStoreDB_LLVMSupport/toolchain_ADT_StringRef.h>
#include <IndexStoreDB_LLVMSupport/toolchain_Support_ConvertUTF.h>
#if defined(_WIN32)
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

using namespace IndexStoreDB;
using namespace index;

// Forward-declare the indexstore symbols

void IndexStoreLibraryProvider::anchor() {}

static IndexStoreLibraryRef loadIndexStoreLibraryFromDLHandle(void *dlHandle, std::string &error);

IndexStoreLibraryRef GlobalIndexStoreLibraryProvider::getLibraryForStorePath(StringRef storePath) {

  // Note: we're using dlsym with RTLD_DEFAULT because we cannot #incldue indexstore.h and indexstore_functions.h
  std::string ignored;
#if defined(_WIN32)
  void* defaultHandle = GetModuleHandleW(NULL);
#else
  void* defaultHandle = RTLD_DEFAULT;
#endif
  return loadIndexStoreLibraryFromDLHandle(defaultHandle, ignored);
}

IndexStoreLibraryRef index::loadIndexStoreLibrary(std::string dylibPath,
                                                  std::string &error) {
#if defined(_WIN32)
  toolchain::SmallVector<toolchain::UTF16, 30> u16Path;
  if (!convertUTF8ToUTF16String(dylibPath, u16Path)) {
    error += "Failed to convert path: " + dylibPath + " to UTF-16";
    return nullptr;
  }
  HMODULE dlHandle = LoadLibraryW((LPCWSTR)u16Path.data());
  if (dlHandle == NULL) {
    error += "Failed to load " + dylibPath + ". Error: " + std::to_string(GetLastError());
    return nullptr;
  }
#else
  auto flags = RTLD_LAZY | RTLD_LOCAL;
#ifdef RTLD_FIRST
  flags |= RTLD_FIRST;
#endif

  void *dlHandle = dlopen(dylibPath.c_str(), flags);
  if (!dlHandle) {
    error = "failed to dlopen indexstore library: ";
    error += dlerror();
    return nullptr;
  }
#endif

  // Intentionally leak the dlhandle; we have no reason to dlclose it and it may be unsafe.
  (void)dlHandle;

  return loadIndexStoreLibraryFromDLHandle(dlHandle, error);
}

static IndexStoreLibraryRef loadIndexStoreLibraryFromDLHandle(void *dlHandle, std::string &error) {
  indexstore_functions_t api;

#if defined(_WIN32)
#define INDEXSTORE_FUNCTION(fn, required) \
  api.fn = (decltype(indexstore_functions_t::fn))GetProcAddress((HMODULE)dlHandle, "indexstore_" #fn); \
  if (!api.fn && required) { \
    error = "indexstore library missing required function indexstore_" #fn; \
    return nullptr; \
  }
#else
#define INDEXSTORE_FUNCTION(fn, required) \
  api.fn = (decltype(indexstore_functions_t::fn))dlsym(dlHandle, "indexstore_" #fn); \
  if (!api.fn && required) { \
    error = "indexstore library missing required function indexstore_" #fn; \
    return nullptr; \
  }
#endif

#include "indexstore_functions.def"

  return std::make_shared<IndexStoreLibrary>(api);
}
