/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 6, 2023.
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
#ifndef RTC_BASE_WIN_GET_ACTIVATION_FACTORY_H_
#define RTC_BASE_WIN_GET_ACTIVATION_FACTORY_H_

#include <winerror.h>

#include "rtc_base/win/hstring.h"

namespace webrtc {

// Provides access to Core WinRT functions which may not be available on
// Windows 7. Loads functions dynamically at runtime to prevent library
// dependencies.

// Callers must check the return value of ResolveCoreWinRTDelayLoad() before
// using these functions.

bool ResolveCoreWinRTDelayload();

HRESULT RoGetActivationFactoryProxy(HSTRING class_id,
                                    const IID& iid,
                                    void** out_factory);

// Retrieves an activation factory for the type specified.
template <typename InterfaceType, wchar_t const* runtime_class_id>
HRESULT GetActivationFactory(InterfaceType** factory) {
  HSTRING class_id_hstring;
  HRESULT hr = CreateHstring(runtime_class_id, wcslen(runtime_class_id),
                             &class_id_hstring);
  if (FAILED(hr))
    return hr;

  hr = RoGetActivationFactoryProxy(class_id_hstring, IID_PPV_ARGS(factory));
  if (FAILED(hr)) {
    DeleteHstring(class_id_hstring);
    return hr;
  }

  return DeleteHstring(class_id_hstring);
}

}  // namespace webrtc

#endif  // RTC_BASE_WIN_GET_ACTIVATION_FACTORY_H_
