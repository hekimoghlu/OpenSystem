/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 6, 2024.
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

#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <stdint.h>

#ifdef __linux__ 

ssize_t getrandom(void *buf, size_t len, unsigned int flags);

void arc4random_buf(void *buf, size_t nbytes);

void arc4random_buf(void *buf, size_t nbytes) {
  while (nbytes > 0) {
    ssize_t actual_nbytes = 0;
    do {
    	actual_nbytes = getrandom(buf, nbytes, 0);
    } while (actual_nbytes == -1 && errno == EINTR);

    if (actual_nbytes == -1) {
      abort();
    }
    
    buf = (uint8_t *)(buf) + actual_nbytes;
    nbytes -= actual_nbytes;
  }
}

#endif
