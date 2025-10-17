/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 29, 2024.
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
 * @file bits/ioctl.h
 * @brief The ioctl() function.
 */

#include <sys/cdefs.h>

__BEGIN_DECLS

/**
 * [ioctl(2)](https://man7.org/linux/man-pages/man2/ioctl.2.html) operates on device files.
 */
int ioctl(int __fd, int __op, ...);

/*
 * Work around unsigned -> signed conversion warnings: many common ioctl
 * constants are unsigned.
 *
 * Since this workaround introduces an overload to ioctl, it's possible that it
 * will break existing code that takes the address of ioctl. If such a breakage
 * occurs, you can work around it by either:
 * - specifying a concrete, correct type for ioctl (whether it be through a cast
 *   in `(int (*)(int, int, ...))ioctl`, creating a temporary variable with the
 *   type of the ioctl you prefer, ...), or
 * - defining BIONIC_IOCTL_NO_SIGNEDNESS_OVERLOAD, which will make the
 *   overloading go away.
 */
#if !defined(BIONIC_IOCTL_NO_SIGNEDNESS_OVERLOAD)
/* enable_if(1) just exists to break overloading ties. */
int ioctl(int __fd, unsigned __op, ...) __overloadable __enable_if(1, "") __RENAME(ioctl);
#endif

__END_DECLS
