/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 12, 2024.
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
#ifndef AOM_AOM_PORTS_AOM_TIMER_H_
#define AOM_AOM_PORTS_AOM_TIMER_H_

#include "config/aom_config.h"

#if CONFIG_OS_SUPPORT

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
/*
 * Win32 specific includes
 */
#undef NOMINMAX
#define NOMINMAX
#undef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
/*
 * POSIX specific includes
 */
#include <sys/time.h>

/* timersub is not provided by msys at this time. */
#ifndef timersub
#define timersub(a, b, result)                       \
  do {                                               \
    (result)->tv_sec = (a)->tv_sec - (b)->tv_sec;    \
    (result)->tv_usec = (a)->tv_usec - (b)->tv_usec; \
    if ((result)->tv_usec < 0) {                     \
      --(result)->tv_sec;                            \
      (result)->tv_usec += 1000000;                  \
    }                                                \
  } while (0)
#endif
#endif

struct aom_usec_timer {
#if defined(_WIN32)
  LARGE_INTEGER begin, end;
#else
  struct timeval begin, end;
#endif
};

static inline void aom_usec_timer_start(struct aom_usec_timer *t) {
#if defined(_WIN32)
  QueryPerformanceCounter(&t->begin);
#else
  gettimeofday(&t->begin, NULL);
#endif
}

static inline void aom_usec_timer_mark(struct aom_usec_timer *t) {
#if defined(_WIN32)
  QueryPerformanceCounter(&t->end);
#else
  gettimeofday(&t->end, NULL);
#endif
}

static inline int64_t aom_usec_timer_elapsed(struct aom_usec_timer *t) {
#if defined(_WIN32)
  LARGE_INTEGER freq, diff;

  diff.QuadPart = t->end.QuadPart - t->begin.QuadPart;

  QueryPerformanceFrequency(&freq);
  return diff.QuadPart * 1000000 / freq.QuadPart;
#else
  struct timeval diff;

  timersub(&t->end, &t->begin, &diff);
  return ((int64_t)diff.tv_sec) * 1000000 + diff.tv_usec;
#endif
}

#else /* CONFIG_OS_SUPPORT = 0*/

/* Empty timer functions if CONFIG_OS_SUPPORT = 0 */
#ifndef timersub
#define timersub(a, b, result)
#endif

struct aom_usec_timer {
  void *dummy;
};

static inline void aom_usec_timer_start(struct aom_usec_timer *t) { (void)t; }

static inline void aom_usec_timer_mark(struct aom_usec_timer *t) { (void)t; }

static inline int aom_usec_timer_elapsed(struct aom_usec_timer *t) {
  (void)t;
  return 0;
}

#endif /* CONFIG_OS_SUPPORT */

#endif  // AOM_AOM_PORTS_AOM_TIMER_H_
