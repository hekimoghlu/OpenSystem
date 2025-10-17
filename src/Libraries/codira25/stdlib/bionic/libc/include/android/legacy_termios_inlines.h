/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 15, 2022.
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

#include <sys/cdefs.h>

// The last bugfixes to <bits/termios_inlines.h> were
// 5da96467a99254c963aef44e75167661d3e02278, so even those these functions were
// in API level 21, ensure that everyone's using the latest versions.
#if __ANDROID_API__ < 28

#include <linux/termios.h>
#include <sys/ioctl.h>
#include <sys/types.h>

#define __BIONIC_TERMIOS_INLINE static __inline
#include <bits/termios_inlines.h>

#endif

#if __ANDROID_API__ < 35

#define __BIONIC_TERMIOS_WINSIZE_INLINE static __inline
#include <bits/termios_winsize_inlines.h>

#endif
