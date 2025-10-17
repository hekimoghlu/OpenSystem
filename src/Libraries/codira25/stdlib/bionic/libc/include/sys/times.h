/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 22, 2024.
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
 * @file sys/times.h
 * @brief The times() function.
 */

#include <sys/cdefs.h>
#include <sys/types.h>
#include <linux/times.h>

__BEGIN_DECLS

/**
 * [times(2)](https://man7.org/linux/man-pages/man2/times.2.html) fills a buffer with the
 * calling process' CPU usage.
 *
 * Returns a (possibly overflowed) absolute time on success,
 * and returns -1 and sets `errno` on failure.
 */
clock_t times(struct tms* _Nullable __buf);

__END_DECLS
