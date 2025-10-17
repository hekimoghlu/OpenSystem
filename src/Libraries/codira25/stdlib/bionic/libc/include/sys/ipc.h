/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 11, 2023.
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
 * @file sys/ipc.h
 * @brief System V IPC.
 */

#include <sys/cdefs.h>
#include <sys/types.h>
#include <linux/ipc.h>

#if defined(__USE_GNU)
#define __key key
#define __seq seq
#endif

#define ipc_perm ipc64_perm

__BEGIN_DECLS

/**
 * [ftok(3)](https://man7.org/linux/man-pages/man3/ftok.3.html) converts a path and id to a
 * System V IPC key.
 *
 * Returns a key on success, and returns -1 and sets `errno` on failure.
 */
key_t ftok(const char* _Nonnull __path, int __id);

__END_DECLS
