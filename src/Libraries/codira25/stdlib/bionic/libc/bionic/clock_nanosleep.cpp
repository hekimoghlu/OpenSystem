/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 16, 2022.
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
#include <time.h>

#include "private/ErrnoRestorer.h"

extern "C" int __clock_nanosleep(clockid_t, int, const timespec*, timespec*);

int clock_nanosleep(clockid_t clock_id, int flags, const timespec* in, timespec* out) {
  if (clock_id == CLOCK_THREAD_CPUTIME_ID) return EINVAL;

  ErrnoRestorer errno_restorer;
  return (__clock_nanosleep(clock_id, flags, in, out) == 0) ? 0 : errno;
}
