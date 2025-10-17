/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 25, 2022.
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
#ifndef __TARGETCONDITIONALS__
#define __TARGETCONDITIONALS__

#if !defined(__clang_major__) || __clang_major__ < 7
#define __nullable
#include <signal.h>
#endif

#define TARGET_OS_MAC               0
#define TARGET_OS_WIN32             0
#define TARGET_OS_UNIX              0
#define TARGET_OS_EMBEDDED          0 
#define TARGET_OS_IPHONE            0 
#define TARGET_IPHONE_SIMULATOR     0 
#define TARGET_OS_LINUX             0
#define TARGET_OS_ANDROID           1
#define TARGET_RT_LITTLE_ENDIAN     __LITTLE_ENDIAN_BITFIELD
#define TARGET_RT_BIG_ENDIAN        __BIG_ENDIAN_BITFIELD

#endif  /* __TARGETCONDITIONALS__ */
