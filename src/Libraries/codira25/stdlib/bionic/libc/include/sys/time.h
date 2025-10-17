/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 26, 2024.
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
#ifndef _SYS_TIME_H_
#define _SYS_TIME_H_

#include <sys/cdefs.h>
#include <sys/types.h>
#include <linux/time.h>

/* POSIX says <sys/time.h> gets you most of <sys/select.h> and may get you all of it. */
#include <sys/select.h>

__BEGIN_DECLS

int gettimeofday(struct timeval* _Nullable __tv, struct timezone* _Nullable __tz);
int settimeofday(const struct timeval* _Nullable __tv, const struct timezone* _Nullable __tz);

int getitimer(int __which, struct itimerval* _Nonnull __current_value);
int setitimer(int __which, const struct itimerval* _Nonnull __new_value, struct itimerval* _Nullable __old_value);

int utimes(const char* _Nonnull __path, const struct timeval __times[_Nullable 2]);

#if defined(__USE_BSD)

#if __BIONIC_AVAILABILITY_GUARD(26)
int futimes(int __fd, const struct timeval __times[_Nullable 2]) __INTRODUCED_IN(26);
int lutimes(const char* _Nonnull __path, const struct timeval __times[_Nullable 2]) __INTRODUCED_IN(26);
#endif /* __BIONIC_AVAILABILITY_GUARD(26) */

#endif

#if defined(__USE_GNU)
/**
 * [futimesat(2)](https://man7.org/linux/man-pages/man2/futimesat.2.html) sets
 * file timestamps.
 *
 * Note: Linux supports `__path` being NULL (in which case `__dir_fd` need not
 * be a directory), allowing futimensat() to be implemented with utimensat().
 * Most callers should just use utimensat() directly, especially on Android
 * where utimensat() has been available for longer than futimesat().
 *
 * Returns 0 on success and -1 and sets `errno` on failure.
 *
 * Available since API level 26.
 */

#if __BIONIC_AVAILABILITY_GUARD(26)
int futimesat(int __dir_fd, const char* __BIONIC_COMPLICATED_NULLNESS __path, const struct timeval __times[_Nullable 2]) __INTRODUCED_IN(26);
#endif /* __BIONIC_AVAILABILITY_GUARD(26) */

#endif

#define timerclear(a)   \
        ((a)->tv_sec = (a)->tv_usec = 0)

#define timerisset(a)    \
        ((a)->tv_sec != 0 || (a)->tv_usec != 0)

#define timercmp(a, b, op)               \
        ((a)->tv_sec == (b)->tv_sec      \
        ? (a)->tv_usec op (b)->tv_usec   \
        : (a)->tv_sec op (b)->tv_sec)

#define timeradd(a, b, res)                           \
    do {                                              \
        (res)->tv_sec  = (a)->tv_sec  + (b)->tv_sec;  \
        (res)->tv_usec = (a)->tv_usec + (b)->tv_usec; \
        if ((res)->tv_usec >= 1000000) {              \
            (res)->tv_usec -= 1000000;                \
            (res)->tv_sec  += 1;                      \
        }                                             \
    } while (0)

#define timersub(a, b, res)                           \
    do {                                              \
        (res)->tv_sec  = (a)->tv_sec  - (b)->tv_sec;  \
        (res)->tv_usec = (a)->tv_usec - (b)->tv_usec; \
        if ((res)->tv_usec < 0) {                     \
            (res)->tv_usec += 1000000;                \
            (res)->tv_sec  -= 1;                      \
        }                                             \
    } while (0)

#define TIMEVAL_TO_TIMESPEC(tv, ts) {     \
    (ts)->tv_sec = (tv)->tv_sec;          \
    (ts)->tv_nsec = (tv)->tv_usec * 1000; \
}
#define TIMESPEC_TO_TIMEVAL(tv, ts) {     \
    (tv)->tv_sec = (ts)->tv_sec;          \
    (tv)->tv_usec = (ts)->tv_nsec / 1000; \
}

__END_DECLS

#endif
