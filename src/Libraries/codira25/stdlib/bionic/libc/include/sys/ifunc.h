/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 22, 2024.
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

/**
 * @file sys/ifunc.h
 * @brief Declarations used for ifunc resolvers. Currently only meaningful for arm64.
 */

__BEGIN_DECLS

#if defined(__aarch64__)

/**
 * Provides information about hardware capabilities to arm64 ifunc resolvers.
 *
 * Prior to API level 30, arm64 ifunc resolvers are passed no arguments.
 *
 * Starting with API level 30, arm64 ifunc resolvers are passed two arguments.
 * The first is a uint64_t whose value is equal to getauxval(AT_HWCAP) | _IFUNC_ARG_HWCAP.
 * The second is a pointer to a data structure of this type.
 *
 * Code that wishes to be compatible with API levels before 30 must call getauxval() itself.
 */
typedef struct __ifunc_arg_t {
  /** Set to sizeof(__ifunc_arg_t). */
  unsigned long _size;

  /** Set to getauxval(AT_HWCAP). */
  unsigned long _hwcap;

  /** Set to getauxval(AT_HWCAP2). */
  unsigned long _hwcap2;
} __ifunc_arg_t;

/**
 * If this bit is set in the first argument to an ifunc resolver, the second argument
 * is a pointer to a data structure of type __ifunc_arg_t.
 *
 * This bit is always set on Android starting with API level 30.
 * This bit is meaningless before API level 30 because ifunc resolvers are not passed any arguments.
 * This bit has no real use on Android, but is included for glibc source compatibility;
 * glibc used this bit to distinguish the case where the ifunc resolver received a single argument,
 * which was an evolutionary stage Android never went through.
 */
#define _IFUNC_ARG_HWCAP (1ULL << 62)

#endif

__END_DECLS
