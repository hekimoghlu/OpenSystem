/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 12, 2022.
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
/* Only include this if we're compiling for the windows platform. */
#ifndef _MSC_VER
#include_next <arm64intr.h>
#else

#ifndef __ARM64INTR_H
#define __ARM64INTR_H

typedef enum
{
  _ARM64_BARRIER_SY    = 0xF,
  _ARM64_BARRIER_ST    = 0xE,
  _ARM64_BARRIER_LD    = 0xD,
  _ARM64_BARRIER_ISH   = 0xB,
  _ARM64_BARRIER_ISHST = 0xA,
  _ARM64_BARRIER_ISHLD = 0x9,
  _ARM64_BARRIER_NSH   = 0x7,
  _ARM64_BARRIER_NSHST = 0x6,
  _ARM64_BARRIER_NSHLD = 0x5,
  _ARM64_BARRIER_OSH   = 0x3,
  _ARM64_BARRIER_OSHST = 0x2,
  _ARM64_BARRIER_OSHLD = 0x1
} _ARM64INTR_BARRIER_TYPE;

#endif /* __ARM64INTR_H */
#endif /* _MSC_VER */
