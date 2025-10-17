/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 31, 2023.
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


#include "libyuv/convert.h"

#include <stdio.h>   // for printf
#include <string.h>  // for memset

int main(int, char**) {
  unsigned char src_i444[640 * 400 * 3];
  unsigned char dst_nv12[640 * 400 * 3 / 2];

  for (size_t i = 0; i < sizeof(src_i444); ++i) {
    src_i444[i] = i & 255;
  }
  memset(dst_nv12, 0, sizeof(dst_nv12));
  libyuv::I444ToNV12(&src_i444[0], 640,              // source Y
                     &src_i444[640 * 400], 640,      // source U
                     &src_i444[640 * 400 * 2], 640,  // source V
                     &dst_nv12[0], 640,              // dest Y
                     &dst_nv12[640 * 400], 640,      // dest UV
                     640, 400);                      // width and height

  int checksum = 0;
  for (size_t i = 0; i < sizeof(dst_nv12); ++i) {
    checksum += dst_nv12[i];
  }
  printf("checksum %x %s\n", checksum, checksum == 0x2ec0c00 ? "PASS" : "FAIL");
  return 0;
}