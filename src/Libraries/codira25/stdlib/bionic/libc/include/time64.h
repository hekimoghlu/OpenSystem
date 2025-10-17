/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 3, 2023.
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
#ifndef TIME64_H
#define TIME64_H

#if defined(__LP64__)

#error Your time_t is already 64-bit.

#else

/* Legacy cruft for LP32 where time_t was 32-bit. */

#include <sys/cdefs.h>
#include <time.h>
#include <stdint.h>

__BEGIN_DECLS

typedef int64_t time64_t;

char* _Nullable asctime64(const struct tm* _Nonnull);
char* _Nullable asctime64_r(const struct tm* _Nonnull, char* _Nonnull);
char* _Nullable ctime64(const time64_t* _Nonnull);
char* _Nullable ctime64_r(const time64_t* _Nonnull, char* _Nonnull);
struct tm* _Nullable gmtime64(const time64_t* _Nonnull);
struct tm* _Nullable gmtime64_r(const time64_t* _Nonnull, struct tm* _Nonnull);
struct tm* _Nullable localtime64(const time64_t* _Nonnull);
struct tm* _Nullable localtime64_r(const time64_t* _Nonnull, struct tm* _Nonnull);
time64_t mktime64(const struct tm* _Nonnull);
time64_t timegm64(const struct tm* _Nonnull);
time64_t timelocal64(const struct tm* _Nonnull);

__END_DECLS

#endif

#endif /* TIME64_H */
