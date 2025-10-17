/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 10, 2025.
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
#ifndef _I386_HW_DEFS_H_
#define _I386_HW_DEFS_H_


#define pmMwaitC1       0x00
#define pmMwaitC2       0x10
#define pmMwaitC3       0x20
#define pmMwaitC4       0x30
#define pmMwaitBrInt 0x1

#define pmBase          0x400
#define pmCtl1          0x04
#define pmCtl2          0x20
#define pmC3Res         0x54
#define pmStatus        0x00
#define msrTSC          0x10

#define cfgAdr          0xCF8
#define cfgDat          0xCFC

#define XeonCapID5      (0x80000000 | (1 << 16) | (30 << 11) | (3 << 8) | 0x98)

#endif /* _I386_HW_DEFS_H_ */
