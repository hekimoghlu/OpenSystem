/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 18, 2023.
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
/*
 *  BLBlockChecksum.c
 *  bless
 *
 *  Created by Shantonu Sen <ssen@apple.com> on Wed Feb 28 2002.
 *  Copyright (c) 2002-2007 Apple Inc. All Rights Reserved.
 *
 *  $Id: BLBlockChecksum.c,v 1.8 2006/02/20 22:49:56 ssen Exp $
 *
 */

#include <sys/types.h>

#include "bless_private.h"

/*
 * Taken from MediaKit. Used to checksum secondary loader
 * presently
 */

uint32_t BLBlockChecksum(const void *buf,uint32_t length)
{
  uint32_t          sum = 0;
  uint32_t          *s = (uint32_t *) buf;
  uint32_t          *t = s + length/4;

  while (s < t) {
    //      rotate 1 bit left and add bytes
    sum = ((sum >> 31) | (sum << 1)) + *s++;
  }
  return sum;
}

