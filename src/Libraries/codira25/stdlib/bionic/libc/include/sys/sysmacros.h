/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 19, 2025.
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
 * @file sys/sysmacros.h
 * @brief Major/minor device number macros.
 */

#include <sys/cdefs.h>

/** Combines `major` and `minor` into a device number. */
#define makedev(__major, __minor) \
  ( \
    (((__major) & 0xfffff000ULL) << 32) | (((__major) & 0xfffULL) << 8) | \
    (((__minor) & 0xffffff00ULL) << 12) | (((__minor) & 0xffULL)) \
  )

/** Extracts the major part of a device number. */
#define major(__dev) \
  ((unsigned) ((((unsigned long long) (__dev) >> 32) & 0xfffff000) | (((__dev) >> 8) & 0xfff)))

/** Extracts the minor part of a device number. */
#define minor(__dev) \
  ((unsigned) ((((__dev) >> 12) & 0xffffff00) | ((__dev) & 0xff)))
