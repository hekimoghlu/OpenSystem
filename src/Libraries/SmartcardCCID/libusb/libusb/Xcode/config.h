/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 18, 2024.
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
#include <AvailabilityMacros.h>

/* Define to the attribute for default visibility. */
#define DEFAULT_VISIBILITY __attribute__ ((visibility ("default")))

/* Define to 1 to enable message logging. */
#define ENABLE_LOGGING 1

/* On 10.12 and later, use newly available clock_*() functions */
#if MAC_OS_X_VERSION_MIN_REQUIRED >= 101200
/* Define to 1 if you have the `clock_gettime' function. */
#define HAVE_CLOCK_GETTIME 1
#endif

/* On 10.6 and later, use newly available pthread_threadid_np() function */
#if MAC_OS_X_VERSION_MIN_REQUIRED >= 1060
/* Define to 1 if you have the 'pthread_threadid_np' function. */
#define HAVE_PTHREAD_THREADID_NP 1
#endif

/* Define to 1 if the system has the type `nfds_t'. */
#define HAVE_NFDS_T 1

/* Define to 1 if you have the <sys/time.h> header file. */
#define HAVE_SYS_TIME_H 1

/* Define to 1 if compiling for a POSIX platform. */
#define PLATFORM_POSIX 1

/* Define to the attribute for enabling parameter checks on printf-like
   functions. */
#define PRINTF_FORMAT(a, b) __attribute__ ((__format__ (__printf__, a, b)))

/* Enable GNU extensions. */
#define _GNU_SOURCE 1
