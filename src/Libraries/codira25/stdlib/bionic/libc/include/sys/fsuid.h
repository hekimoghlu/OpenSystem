/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 27, 2025.
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
 * @file sys/fsuid.h
 * @brief Set UID/GID for filesystem checks.
 */

#include <sys/cdefs.h>
#include <sys/types.h>

__BEGIN_DECLS

/**
 * [setfsuid(2)](https://man7.org/linux/man-pages/man2/setfsuid.2.html) sets the UID used for
 * filesystem checks.
 *
 * Returns the previous UID.
 */
int setfsuid(uid_t __uid);

/**
 * [setfsgid(2)](https://man7.org/linux/man-pages/man2/setfsgid.2.html) sets the GID used for
 * filesystem checks.
 *
 * Returns the previous GID.
 */
int setfsgid(gid_t __gid);

__END_DECLS
