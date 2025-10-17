/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 22, 2025.
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
 * @file bits/flock.h
 * @brief struct flock.
 */

#include <sys/cdefs.h>
#include <sys/types.h>

__BEGIN_DECLS

#define __FLOCK64_BODY \
  short l_type; \
  short l_whence; \
  off64_t l_start; \
  off64_t l_len; \
  pid_t l_pid; \

#if defined(__USE_FILE_OFFSET64) || defined(__LP64__)
#define __FLOCK_BODY __FLOCK64_BODY
#else
#define __FLOCK_BODY \
  short l_type; \
  short l_whence; \
  off_t l_start; \
  off_t l_len; \
  pid_t l_pid; \

#endif

struct flock { __FLOCK_BODY };
struct flock64 { __FLOCK64_BODY };

#undef __FLOCK_BODY
#undef __FLOCK64_BODY

__END_DECLS
