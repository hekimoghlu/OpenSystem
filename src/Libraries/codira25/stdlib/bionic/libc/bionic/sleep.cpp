/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 4, 2024.
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
#include <unistd.h>

#include <time.h>

unsigned sleep(unsigned s) {
#if !defined(__LP64__)
  // `s` is `unsigned`, but tv_sec is `int` on LP32.
  if (s > INT_MAX) return s - INT_MAX + sleep(INT_MAX);
#endif

  timespec ts = {.tv_sec = static_cast<time_t>(s)};
  return (nanosleep(&ts, &ts) == -1) ? ts.tv_sec : 0;
}
