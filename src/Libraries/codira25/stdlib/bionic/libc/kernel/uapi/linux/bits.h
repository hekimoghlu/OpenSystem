/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 7, 2023.
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
#ifndef _UAPI_LINUX_BITS_H
#define _UAPI_LINUX_BITS_H
#define __GENMASK(h,l) (((~_UL(0)) - (_UL(1) << (l)) + 1) & (~_UL(0) >> (__BITS_PER_LONG - 1 - (h))))
#define __GENMASK_ULL(h,l) (((~_ULL(0)) - (_ULL(1) << (l)) + 1) & (~_ULL(0) >> (__BITS_PER_LONG_LONG - 1 - (h))))
#define __GENMASK_U128(h,l) ((_BIT128((h)) << 1) - (_BIT128(l)))
#endif
