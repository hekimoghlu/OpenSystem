/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 23, 2024.
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
#include "rtc_base/win/get_activation_factory.h"

#include <libloaderapi.h>
#include <roapi.h>

namespace {

FARPROC LoadComBaseFunction(const char* function_name) {
  static HMODULE const handle =
      ::LoadLibraryExW(L"combase.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
  return handle ? ::GetProcAddress(handle, function_name) : nullptr;
}

decltype(&::RoGetActivationFactory) GetRoGetActivationFactoryFunction() {
  static decltype(&::RoGetActivationFactory) const function =
      reinterpret_cast<decltype(&::RoGetActivationFactory)>(
          LoadComBaseFunction("RoGetActivationFactory"));
  return function;
}

}  // namespace

namespace webrtc {

bool ResolveCoreWinRTDelayload() {
  return GetRoGetActivationFactoryFunction() &&
         ResolveCoreWinRTStringDelayload();
}

HRESULT RoGetActivationFactoryProxy(HSTRING class_id,
                                    const IID& iid,
                                    void** out_factory) {
  auto get_factory_func = GetRoGetActivationFactoryFunction();
  if (!get_factory_func)
    return E_FAIL;
  return get_factory_func(class_id, iid, out_factory);
}

}  // namespace webrtc
