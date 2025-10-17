/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 27, 2024.
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
#ifndef _ARM_ARCH_H
#define _ARM_ARCH_H

#if defined (__arm__) || defined (__arm64__)

/* Collect the __ARM_ARCH_*__ compiler flags into something easier to use. */
#if defined (__ARM_ARCH_7A__) || defined (__ARM_ARCH_7S__) || defined (__ARM_ARCH_7F__) || defined (__ARM_ARCH_7K__)
#define _ARM_ARCH_7
#endif

#if defined (_ARM_ARCH_7) || defined (__ARM_ARCH_6K__) || defined (__ARM_ARCH_6ZK__)
#define _ARM_ARCH_6K
#endif

#if defined (_ARM_ARCH_7) || defined (__ARM_ARCH_6Z__) || defined (__ARM_ARCH_6ZK__)
#define _ARM_ARCH_6Z
#endif

#if defined (__ARM_ARCH_6__) || defined (__ARM_ARCH_6J__) || \
        defined (_ARM_ARCH_6Z) || defined (_ARM_ARCH_6K)
#define _ARM_ARCH_6
#endif

#if defined (_ARM_ARCH_6) || defined (__ARM_ARCH_5E__) || \
        defined (__ARM_ARCH_5TE__) || defined (__ARM_ARCH_5TEJ__)
#define _ARM_ARCH_5E
#endif

#if defined (_ARM_ARCH_5E) || defined (__ARM_ARCH_5__) || \
        defined (__ARM_ARCH_5T__)
#define _ARM_ARCH_5
#endif

#if defined (_ARM_ARCH_5) || defined (__ARM_ARCH_4T__)
#define _ARM_ARCH_4T
#endif

#if defined (_ARM_ARCH_4T) || defined (__ARM_ARCH_4__)
#define _ARM_ARCH_4
#endif

#endif /* defined (__arm__) || defined (__arm64__) */

#endif
