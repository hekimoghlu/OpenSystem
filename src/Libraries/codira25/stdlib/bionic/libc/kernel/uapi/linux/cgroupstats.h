/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 12, 2025.
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
#ifndef _LINUX_CGROUPSTATS_H
#define _LINUX_CGROUPSTATS_H
#include <linux/types.h>
#include <linux/taskstats.h>
struct cgroupstats {
  __u64 nr_sleeping;
  __u64 nr_running;
  __u64 nr_stopped;
  __u64 nr_uninterruptible;
  __u64 nr_io_wait;
};
enum {
  CGROUPSTATS_CMD_UNSPEC = __TASKSTATS_CMD_MAX,
  CGROUPSTATS_CMD_GET,
  CGROUPSTATS_CMD_NEW,
  __CGROUPSTATS_CMD_MAX,
};
#define CGROUPSTATS_CMD_MAX (__CGROUPSTATS_CMD_MAX - 1)
enum {
  CGROUPSTATS_TYPE_UNSPEC = 0,
  CGROUPSTATS_TYPE_CGROUP_STATS,
  __CGROUPSTATS_TYPE_MAX,
};
#define CGROUPSTATS_TYPE_MAX (__CGROUPSTATS_TYPE_MAX - 1)
enum {
  CGROUPSTATS_CMD_ATTR_UNSPEC = 0,
  CGROUPSTATS_CMD_ATTR_FD,
  __CGROUPSTATS_CMD_ATTR_MAX,
};
#define CGROUPSTATS_CMD_ATTR_MAX (__CGROUPSTATS_CMD_ATTR_MAX - 1)
#endif
