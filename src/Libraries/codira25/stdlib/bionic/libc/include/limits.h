/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 2, 2023.
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
#pragma once

/**
 * @file limits.h
 * @brief Constants relating to implementation limits.
 *
 * This file is included via `#include_next` from the clang header of the same
 * name that provides all the limits that the compiler is responsible for,
 * primarily those relating to integer types defined by the C standard.
 * This file defines the additional limits defined by POSIX.
 */

/*
 * The Android build system has bionic _before_ the clang headers,
 * so although the claim above that clang does an `#include_next`
 * of this file is true for the NDK, it's not true for the OS,
 * and we need to paper over that difference here until/unless
 * the OS build changes.
 */
#if __has_include_next(<limits.h>)
#include_next <limits.h>
#endif

#include <sys/cdefs.h>

/* Historically bionic exposed the content of <float.h> from <limits.h> and <sys/limits.h> too. */
#include <float.h>

/* Many of the POSIX limits come from the kernel. */
#include <linux/limits.h>

/** Maximum number of positional arguments in a printf()/scanf() format string. */
#define NL_ARGMAX 9
/** Maximum number of bytes in a $LANG name. */
#define NL_LANGMAX 14
/** Irrelevant with Android's <nl_types.h>. */
#define NL_MSGMAX 32767
/** Obsolete; removed from POSIX. */
#define NL_NMAX 1
/** Irrelevant with Android's <nl_types.h>. */
#define NL_SETMAX 255
/** Irrelevant with Android's <nl_types.h>. */
#define NL_TEXTMAX 255

/** Obsolete; removed from POSIX. */
#define PASS_MAX 128
/** Obsolete; removed from POSIX. */
#define TMP_MAX 308915776

/** Number of bits in a `long` (POSIX). */
#if __LP64__
#define LONG_BIT 64
#else
#define LONG_BIT 32
#endif
/** Number of bits in a "word" of `int` (POSIX). */
#define WORD_BIT 32

/** Maximum value of a uid_t. */
#define UID_MAX UINT_MAX
/** Maximum value of a gid_t. */
#define GID_MAX UINT_MAX
/** Maximum value of a size_t. */
#define SIZE_T_MAX ULONG_MAX
/** Maximum value of a ssize_t. */
#define SSIZE_MAX LONG_MAX

/**
 * POSIX 2024's name for NSIG.
 * See the NSIG documentation for an explanation and warnings.
 */
#define NSIG_MAX 65

/** Maximum number of bytes in a multibyte character. */
#define MB_LEN_MAX 4

/** Default process priority. */
#define NZERO 20

/** Maximum number of struct iovec that can be passed in a single readv()/writev(). */
#define IOV_MAX 1024

/** Maximum value for a semaphore. */
#define SEM_VALUE_MAX 0x3fffffff

/** Do not use: prefer getline() or asprintf() rather than hard-coding an arbitrary size. */
#define LINE_MAX _POSIX2_LINE_MAX

/* POSIX says these belong in <unistd.h> but BSD has some in <limits.h>. */
#include <bits/posix_limits.h>

/** Maximum length of a hostname returned by gethostname(). */
#define HOST_NAME_MAX _POSIX_HOST_NAME_MAX

/** Maximum length of a login name. */
#define LOGIN_NAME_MAX 256

/** Maximum length of terminal device name. */
#define TTY_NAME_MAX 32

/** Maximum number of attempts to destroy thread-specific data when a thread exits. */
#define PTHREAD_DESTRUCTOR_ITERATIONS _POSIX_THREAD_DESTRUCTOR_ITERATIONS

/**
 * The number of calls to pthread_key_create() without intervening calls to
 * pthread_key_delete() that are guaranteed to succeed. See pthread_key_create()
 * for more details and ways to avoid hitting this limit.
 */
#define PTHREAD_KEYS_MAX 128

/** bionic has no fixed limit on the number of threads. */
#undef PTHREAD_THREADS_MAX
