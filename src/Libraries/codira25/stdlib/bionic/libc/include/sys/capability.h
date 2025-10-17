/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 1, 2022.
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
 * @file sys/capability.h
 * @brief Capabilities.
 */

#include <sys/cdefs.h>
#include <linux/capability.h>

__BEGIN_DECLS

/**
 * [capget(2)](https://man7.org/linux/man-pages/man2/capget.2.html) gets the calling
 * thread's capabilities.
 *
 * Returns 0 on success, and returns -1 and sets `errno` on failure.
 */
int capget(cap_user_header_t _Nonnull __hdr_ptr, cap_user_data_t _Nullable __data_ptr);

/**
 * [capset(2)](https://man7.org/linux/man-pages/man2/capset.2.html) sets the calling
 * thread's capabilities.
 *
 * Returns 0 on success, and returns -1 and sets `errno` on failure.
 */
int capset(cap_user_header_t _Nonnull __hdr_ptr, const cap_user_data_t _Nullable __data_ptr);

__END_DECLS
