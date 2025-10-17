/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 8, 2023.
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
 * @file sys/prctl.h
 * @brief Process-specific operations.
 */

#include <sys/cdefs.h>

#include <linux/prctl.h>

__BEGIN_DECLS

/**
 * [prctl(2)](https://man7.org/linux/man-pages/man2/prctl.2.html) performs a variety of
 * operations based on the `PR_` constant passed as the first argument.
 *
 * Returns -1 and sets `errno` on failure; success values vary by option.
 */
int prctl(int __op, ...);

__END_DECLS
