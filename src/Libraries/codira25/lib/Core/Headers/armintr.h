/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 3, 2024.
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
#include_next <armintr.h>
#else

#ifndef __ARMINTR_H
#define __ARMINTR_H

typedef enum
{
  _ARM_BARRIER_SY    = 0xF,
  _ARM_BARRIER_ST    = 0xE,
  _ARM_BARRIER_ISH   = 0xB,
  _ARM_BARRIER_ISHST = 0xA,
  _ARM_BARRIER_NSH   = 0x7,
  _ARM_BARRIER_NSHST = 0x6,
  _ARM_BARRIER_OSH   = 0x3,
  _ARM_BARRIER_OSHST = 0x2
} _ARMINTR_BARRIER_TYPE;

#endif /* __ARMINTR_H */
#endif /* _MSC_VER */
