/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 7, 2022.
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
#ifndef RTC_BASE_WIN_SCOPED_COM_INITIALIZER_H_
#define RTC_BASE_WIN_SCOPED_COM_INITIALIZER_H_

#include <comdef.h>

namespace webrtc {

// Initializes COM in the constructor (STA or MTA), and uninitializes COM in the
// destructor. Taken from base::win::ScopedCOMInitializer.
//
// WARNING: This should only be used once per thread, ideally scoped to a
// similar lifetime as the thread itself.  You should not be using this in
// random utility functions that make COM calls; instead ensure that these
// functions are running on a COM-supporting thread!
// See https://msdn.microsoft.com/en-us/library/ms809971.aspx for details.
class ScopedCOMInitializer {
 public:
  // Enum value provided to initialize the thread as an MTA instead of STA.
  // There are two types of apartments, Single Threaded Apartments (STAs)
  // and Multi Threaded Apartments (MTAs). Within a given process there can
  // be multiple STAâ€™s but there is only one MTA. STA is typically used by
  // "GUI applications" and MTA by "worker threads" with no UI message loop.
  enum SelectMTA { kMTA };

  // Constructor for STA initialization.
  ScopedCOMInitializer();

  // Constructor for MTA initialization.
  explicit ScopedCOMInitializer(SelectMTA mta);

  ~ScopedCOMInitializer();

  ScopedCOMInitializer(const ScopedCOMInitializer&) = delete;
  ScopedCOMInitializer& operator=(const ScopedCOMInitializer&) = delete;

  bool Succeeded() { return SUCCEEDED(hr_); }

 private:
  void Initialize(COINIT init);

  HRESULT hr_;
};

}  // namespace webrtc

#endif  // RTC_BASE_WIN_SCOPED_COM_INITIALIZER_H_
