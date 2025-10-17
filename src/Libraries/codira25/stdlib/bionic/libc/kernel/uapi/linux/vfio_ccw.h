/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 27, 2025.
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
#ifndef _VFIO_CCW_H_
#define _VFIO_CCW_H_
#include <linux/types.h>
struct ccw_io_region {
#define ORB_AREA_SIZE 12
  __u8 orb_area[ORB_AREA_SIZE];
#define SCSW_AREA_SIZE 12
  __u8 scsw_area[SCSW_AREA_SIZE];
#define IRB_AREA_SIZE 96
  __u8 irb_area[IRB_AREA_SIZE];
  __u32 ret_code;
} __attribute__((__packed__));
#define VFIO_CCW_ASYNC_CMD_HSCH (1 << 0)
#define VFIO_CCW_ASYNC_CMD_CSCH (1 << 1)
struct ccw_cmd_region {
  __u32 command;
  __u32 ret_code;
} __attribute__((__packed__));
struct ccw_schib_region {
#define SCHIB_AREA_SIZE 52
  __u8 schib_area[SCHIB_AREA_SIZE];
} __attribute__((__packed__));
struct ccw_crw_region {
  __u32 crw;
  __u32 pad;
} __attribute__((__packed__));
#endif
