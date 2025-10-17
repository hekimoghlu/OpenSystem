/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 4, 2022.
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
 * @file alloca.h
 * @brief Allocate space on the stack.
 */

#include <sys/cdefs.h>

/**
 * [alloca(3)](https://man7.org/linux/man-pages/man3/alloca.3.html) allocates space on the stack.
 *
 * New code should not use alloca because it cannot report failure.
 * Use regular heap allocation instead.
 *
 * @return a pointer to the space on success, but has undefined behavior on failure.
 */
#define alloca(size)   __builtin_alloca(size)
