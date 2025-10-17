/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 18, 2022.
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
 * @file bits/seek_constants.h
 * @brief The `SEEK_` constants.
 */

#include <sys/cdefs.h>

/** Seek to an absolute offset. */
#define SEEK_SET 0
/** Seek relative to the current offset. */
#define SEEK_CUR 1
/** Seek relative to the end of the file. */
#define SEEK_END 2

#if defined(__USE_GNU)

/**
 * Seek to the first data (non-hole) location in the file
 * greater than or equal to the given offset.
 *
 * See [lseek(2)](https://man7.org/linux/man-pages/man2/lseek.2.html).
 */
#define SEEK_DATA 3

/**
 * Seek to the first hole (non-data) location in the file
 * greater than or equal to the given offset.
 *
 * See [lseek(2)](https://man7.org/linux/man-pages/man2/lseek.2.html).
 */
#define SEEK_HOLE 4

#endif
