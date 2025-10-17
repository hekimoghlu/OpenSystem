/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 26, 2021.
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
#ifndef _XT_TIME_H
#define _XT_TIME_H 1
#include <linux/types.h>
struct xt_time_info {
  __u32 date_start;
  __u32 date_stop;
  __u32 daytime_start;
  __u32 daytime_stop;
  __u32 monthdays_match;
  __u8 weekdays_match;
  __u8 flags;
};
enum {
  XT_TIME_LOCAL_TZ = 1 << 0,
  XT_TIME_CONTIGUOUS = 1 << 1,
  XT_TIME_ALL_MONTHDAYS = 0xFFFFFFFE,
  XT_TIME_ALL_WEEKDAYS = 0xFE,
  XT_TIME_MIN_DAYTIME = 0,
  XT_TIME_MAX_DAYTIME = 24 * 60 * 60 - 1,
};
#define XT_TIME_ALL_FLAGS (XT_TIME_LOCAL_TZ | XT_TIME_CONTIGUOUS)
#endif
