/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 2, 2024.
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
 * @file bits/timespec.h
 * @brief The `timespec` struct.
 */

#include <sys/cdefs.h>
#include <sys/types.h>

/*
 * This file is used to include timespec definition without introducing the whole
 * <linux/time.h>, <sys/time.h> or <time.h>.
 */
#ifndef _STRUCT_TIMESPEC
#define _STRUCT_TIMESPEC
/** Represents a time. */
struct timespec {
  /** Number of seconds. */
  time_t tv_sec;
  /** Number of nanoseconds. Must be less than 1,000,000,000. */
  long tv_nsec;
};
#endif
