/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 22, 2024.
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
#include "rtc_base/win/scoped_com_initializer.h"

#include "rtc_base/checks.h"
#include "rtc_base/logging.h"

namespace webrtc {

ScopedCOMInitializer::ScopedCOMInitializer() {
  RTC_DLOG(LS_INFO) << "Single-Threaded Apartment (STA) COM thread";
  Initialize(COINIT_APARTMENTTHREADED);
}

// Constructor for MTA initialization.
ScopedCOMInitializer::ScopedCOMInitializer(SelectMTA mta) {
  RTC_DLOG(LS_INFO) << "Multi-Threaded Apartment (MTA) COM thread";
  Initialize(COINIT_MULTITHREADED);
}

ScopedCOMInitializer::~ScopedCOMInitializer() {
  if (Succeeded()) {
    CoUninitialize();
  }
}

void ScopedCOMInitializer::Initialize(COINIT init) {
  // Initializes the COM library for use by the calling thread, sets the
  // thread's concurrency model, and creates a new apartment for the thread
  // if one is required. CoInitializeEx must be called at least once, and is
  // usually called only once, for each thread that uses the COM library.
  hr_ = CoInitializeEx(NULL, init);
  RTC_CHECK_NE(RPC_E_CHANGED_MODE, hr_)
      << "Invalid COM thread model change (MTA->STA)";
  // Multiple calls to CoInitializeEx by the same thread are allowed as long
  // as they pass the same concurrency flag, but subsequent valid calls
  // return S_FALSE. To close the COM library gracefully on a thread, each
  // successful call to CoInitializeEx, including any call that returns
  // S_FALSE, must be balanced by a corresponding call to CoUninitialize.
  if (hr_ == S_OK) {
    RTC_DLOG(LS_INFO)
        << "The COM library was initialized successfully on this thread";
  } else if (hr_ == S_FALSE) {
    RTC_DLOG(LS_WARNING)
        << "The COM library is already initialized on this thread";
  }
}

}  // namespace webrtc
