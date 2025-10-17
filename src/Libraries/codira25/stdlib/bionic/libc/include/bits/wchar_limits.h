/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 20, 2024.
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
 * @file bits/wchar_limits.h
 * @brief `wchar_t` limits.
 */

#include <sys/cdefs.h>

/** The maximum value of a `wchar_t`. */
#define WCHAR_MAX __WCHAR_MAX__

/* As of 3.4, clang still doesn't define __WCHAR_MIN__. */
#if defined(__WCHAR_UNSIGNED__)
/** The minimum value of a `wchar_t`. */
#  define WCHAR_MIN L'\0'
#else
/** The minimum value of a `wchar_t`. */
#  define WCHAR_MIN (-(WCHAR_MAX) - 1)
#endif
