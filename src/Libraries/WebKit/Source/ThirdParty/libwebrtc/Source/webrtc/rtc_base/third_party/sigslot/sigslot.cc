/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 1, 2023.
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

// sigslot.h: Signal/Slot classes
//
// Written by Sarah Thompson (sarah@telergy.com) 2002.
//
// License: Public domain. You are free to use this code however you like, with
// the proviso that the author takes on no responsibility or liability for any
// use.

#include "rtc_base/third_party/sigslot/sigslot.h"

namespace sigslot {

#ifdef _SIGSLOT_HAS_POSIX_THREADS

pthread_mutex_t* multi_threaded_global::get_mutex() {
  static pthread_mutex_t g_mutex = PTHREAD_MUTEX_INITIALIZER;
  return &g_mutex;
}

#endif  // _SIGSLOT_HAS_POSIX_THREADS

}  // namespace sigslot
