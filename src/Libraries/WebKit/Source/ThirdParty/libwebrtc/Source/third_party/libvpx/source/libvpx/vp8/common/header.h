/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 25, 2022.
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
#ifndef VPX_VP8_COMMON_HEADER_H_
#define VPX_VP8_COMMON_HEADER_H_

#ifdef __cplusplus
extern "C" {
#endif

/* 24 bits total */
typedef struct {
  unsigned int type : 1;
  unsigned int version : 3;
  unsigned int show_frame : 1;

  /* Allow 2^20 bytes = 8 megabits for first partition */

  unsigned int first_partition_length_in_bytes : 19;

#ifdef PACKET_TESTING
  unsigned int frame_number;
  unsigned int update_gold : 1;
  unsigned int uses_gold : 1;
  unsigned int update_last : 1;
  unsigned int uses_last : 1;
#endif

} VP8_HEADER;

#ifdef PACKET_TESTING
#define VP8_HEADER_SIZE 8
#else
#define VP8_HEADER_SIZE 3
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_COMMON_HEADER_H_
