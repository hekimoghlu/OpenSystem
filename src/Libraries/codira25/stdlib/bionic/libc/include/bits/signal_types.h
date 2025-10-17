/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 22, 2024.
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

#include <sys/cdefs.h>

#include <limits.h>
#include <linux/signal.h>
#include <sys/types.h>

/**
 * The highest kernel-supported signal number, plus one.
 *
 * In theory this is useful for declaring an array with an entry for each signal.
 * In practice, that's less useful than it seems because of the real-time
 * signals and the reserved signals,
 * and the sig2str() and str2sig() functions cover the most common use case
 * of translating between signal numbers and signal names.
 *
 * Note also that although sigset_t and sigset64_t are the same type on LP64,
 * on ILP32 only sigset64_t is large enough to refer to the upper 32 signals.
 * NSIG does _not_ tell you anything about what can be used with sigset_t.
 *
 * See the
 * (32-bit ABI bugs)[https://android.googlesource.com/platform/bionic/+/main/docs/32-bit-abi.md#is-too-small-for-real_time-signals]
 * documentation.
 */
#define NSIG 65
/** A traditional alternative name for NSIG. */
#define _NSIG 65

/*
 * We rewrite the kernel's _NSIG to _KERNEL__NSIG
 * (because the kernel values are off by one from the userspace values),
 * but the kernel <asm/signal.h> headers define SIGRTMAX in terms of
 * _KERNEL__NSIG (or _NSIG, in the original kernel source),
 * so we need to provide a definition here.
 * (Ideally our uapi header rewriter would just hard-code _KERNEL__NSIG to 64.)
 */
#ifndef _KERNEL__NSIG
#define _KERNEL__NSIG 64
#endif

typedef int sig_atomic_t;

typedef __sighandler_t sig_t; /* BSD compatibility. */
typedef __sighandler_t sighandler_t; /* glibc compatibility. */

#if defined(__LP64__)
/**
 * The kernel LP64 sigset_t is large enough to support all signals;
 * this typedef is just for source compatibility with code that uses
 * real-time signals on ILP32.
 *
 * See the
 * (32-bit ABI bugs)[https://android.googlesource.com/platform/bionic/+/main/docs/32-bit-abi.md#is-too-small-for-real_time-signals]
 * documentation.
 */
typedef sigset_t sigset64_t;
#else
/**
 * The ILP32 sigset_t is only 32 bits, so we need a 64-bit sigset64_t
 * and associated functions to be able to support the real-time signals.
 *
 * See the
 * (32-bit ABI bugs)[https://android.googlesource.com/platform/bionic/+/main/docs/32-bit-abi.md#is-too-small-for-real_time-signals]
 * documentation.
 */
typedef struct { unsigned long __bits[64/(8*sizeof(long))]; } sigset64_t;
#endif

/* The kernel's struct sigaction doesn't match the POSIX one,
 * so we define struct sigaction ourselves. */

#if defined(__LP64__)

#define __SIGACTION_BODY \
  int sa_flags; \
  union { \
    sighandler_t sa_handler; \
    void (*sa_sigaction)(int, struct siginfo*, void*); \
  }; \
  sigset_t sa_mask; \
  void (*sa_restorer)(void); \

/**
 * Used with sigaction().
 *
 * On LP64, this supports all signals including real-time signals.
 * On ILP32, this only supports the first 32 signals.
 *
 * See the
 * (32-bit ABI bugs)[https://android.googlesource.com/platform/bionic/+/main/docs/32-bit-abi.md#is-too-small-for-real_time-signals]
 * documentation.
 */
struct sigaction { __SIGACTION_BODY };
/**
 * Used with sigaction64().
 *
 * On LP64, a synonym for struct sigaction for source compatibility with ILP32.
 * On ILP32, this is needed to support all signals including real-time signals
 * because struct sigaction only supports the first 32 signals.
 *
 * See the
 * (32-bit ABI bugs)[https://android.googlesource.com/platform/bionic/+/main/docs/32-bit-abi.md#is-too-small-for-real_time-signals]
 * documentation.
 */
struct sigaction64 { __SIGACTION_BODY };

#undef __SIGACTION_BODY

#else

/* The arm32 kernel headers pollute the namespace with these,
 * but our header scrubber doesn't know how to remove #defines. */
#undef sa_handler
#undef sa_sigaction

/**
 * Used with sigaction().
 *
 * On LP64, this supports all signals including real-time signals.
 * On ILP32, this only supports the first 32 signals.
 *
 * See the
 * (32-bit ABI bugs)[https://android.googlesource.com/platform/bionic/+/main/docs/32-bit-abi.md#is-too-small-for-real_time-signals]
 * documentation.
 */
struct sigaction {
  union {
    sighandler_t sa_handler;
    void (*sa_sigaction)(int, struct siginfo*, void*);
  };
  sigset_t sa_mask;
  int sa_flags;
  void (*sa_restorer)(void);
};

/**
 * Used with sigaction64().
 *
 * On LP64, a synonym for struct sigaction for source compatibility with ILP32.
 * On ILP32, this is needed to support all signals including real-time signals
 * because struct sigaction only supports the first 32 signals.
 *
 * See the
 * (32-bit ABI bugs)[https://android.googlesource.com/platform/bionic/+/main/docs/32-bit-abi.md#is-too-small-for-real_time-signals]
 * documentation.
 */
struct sigaction64 {
  union {
    sighandler_t sa_handler;
    void (*sa_sigaction)(int, struct siginfo*, void*);
  };
  int sa_flags;
  void (*sa_restorer)(void);
  sigset64_t sa_mask;
};

#endif
