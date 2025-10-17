/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 30, 2024.
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
#include "modules/audio_device/linux/latebindingsymboltable_linux.h"

#include "absl/strings/string_view.h"
#include "rtc_base/logging.h"

#ifdef WEBRTC_LINUX
#include <dlfcn.h>
#endif

namespace webrtc {
namespace adm_linux {

inline static const char* GetDllError() {
#ifdef WEBRTC_LINUX
  char* err = dlerror();
  if (err) {
    return err;
  } else {
    return "No error";
  }
#else
#error Not implemented
#endif
}

DllHandle InternalLoadDll(absl::string_view dll_name) {
#ifdef WEBRTC_LINUX
  DllHandle handle = dlopen(std::string(dll_name).c_str(), RTLD_NOW);
#else
#error Not implemented
#endif
  if (handle == kInvalidDllHandle) {
    RTC_LOG(LS_WARNING) << "Can't load " << dll_name << " : " << GetDllError();
  }
  return handle;
}

void InternalUnloadDll(DllHandle handle) {
#ifdef WEBRTC_LINUX
// TODO(pbos): Remove this dlclose() exclusion when leaks and suppressions from
// here are gone (or AddressSanitizer can display them properly).
//
// Skip dlclose() on AddressSanitizer as leaks including this module in the
// stack trace gets displayed as <unknown module> instead of the actual library
// -> it can not be suppressed.
// https://code.google.com/p/address-sanitizer/issues/detail?id=89
#if !defined(ADDRESS_SANITIZER)
  if (dlclose(handle) != 0) {
    RTC_LOG(LS_ERROR) << GetDllError();
  }
#endif  // !defined(ADDRESS_SANITIZER)
#else
#error Not implemented
#endif
}

static bool LoadSymbol(DllHandle handle,
                       absl::string_view symbol_name,
                       void** symbol) {
#ifdef WEBRTC_LINUX
  *symbol = dlsym(handle, std::string(symbol_name).c_str());
  char* err = dlerror();
  if (err) {
    RTC_LOG(LS_ERROR) << "Error loading symbol " << symbol_name << " : " << err;
    return false;
  } else if (!*symbol) {
    RTC_LOG(LS_ERROR) << "Symbol " << symbol_name << " is NULL";
    return false;
  }
  return true;
#else
#error Not implemented
#endif
}

// This routine MUST assign SOME value for every symbol, even if that value is
// NULL, or else some symbols may be left with uninitialized data that the
// caller may later interpret as a valid address.
bool InternalLoadSymbols(DllHandle handle,
                         int num_symbols,
                         const char* const symbol_names[],
                         void* symbols[]) {
#ifdef WEBRTC_LINUX
  // Clear any old errors.
  dlerror();
#endif
  for (int i = 0; i < num_symbols; ++i) {
    if (!LoadSymbol(handle, symbol_names[i], &symbols[i])) {
      return false;
    }
  }
  return true;
}

}  // namespace adm_linux
}  // namespace webrtc
