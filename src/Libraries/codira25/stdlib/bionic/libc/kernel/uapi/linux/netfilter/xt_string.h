/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 18, 2024.
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
#ifndef _XT_STRING_H
#define _XT_STRING_H
#include <linux/types.h>
#define XT_STRING_MAX_PATTERN_SIZE 128
#define XT_STRING_MAX_ALGO_NAME_SIZE 16
enum {
  XT_STRING_FLAG_INVERT = 0x01,
  XT_STRING_FLAG_IGNORECASE = 0x02
};
struct xt_string_info {
  __u16 from_offset;
  __u16 to_offset;
  char algo[XT_STRING_MAX_ALGO_NAME_SIZE];
  char pattern[XT_STRING_MAX_PATTERN_SIZE];
  __u8 patlen;
  union {
    struct {
      __u8 invert;
    } v0;
    struct {
      __u8 flags;
    } v1;
  } u;
  struct ts_config __attribute__((aligned(8))) * config;
};
#endif
