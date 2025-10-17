/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 31, 2025.
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
#ifndef _UAPI_REGULATOR_H
#define _UAPI_REGULATOR_H
#include <stdint.h>
#define REGULATOR_EVENT_UNDER_VOLTAGE 0x01
#define REGULATOR_EVENT_OVER_CURRENT 0x02
#define REGULATOR_EVENT_REGULATION_OUT 0x04
#define REGULATOR_EVENT_FAIL 0x08
#define REGULATOR_EVENT_OVER_TEMP 0x10
#define REGULATOR_EVENT_FORCE_DISABLE 0x20
#define REGULATOR_EVENT_VOLTAGE_CHANGE 0x40
#define REGULATOR_EVENT_DISABLE 0x80
#define REGULATOR_EVENT_PRE_VOLTAGE_CHANGE 0x100
#define REGULATOR_EVENT_ABORT_VOLTAGE_CHANGE 0x200
#define REGULATOR_EVENT_PRE_DISABLE 0x400
#define REGULATOR_EVENT_ABORT_DISABLE 0x800
#define REGULATOR_EVENT_ENABLE 0x1000
#define REGULATOR_EVENT_UNDER_VOLTAGE_WARN 0x2000
#define REGULATOR_EVENT_OVER_CURRENT_WARN 0x4000
#define REGULATOR_EVENT_OVER_VOLTAGE_WARN 0x8000
#define REGULATOR_EVENT_OVER_TEMP_WARN 0x10000
#define REGULATOR_EVENT_WARN_MASK 0x1E000
struct reg_genl_event {
  char reg_name[32];
  uint64_t event;
};
enum {
  REG_GENL_ATTR_UNSPEC,
  REG_GENL_ATTR_EVENT,
  __REG_GENL_ATTR_MAX,
};
#define REG_GENL_ATTR_MAX (__REG_GENL_ATTR_MAX - 1)
enum {
  REG_GENL_CMD_UNSPEC,
  REG_GENL_CMD_EVENT,
  __REG_GENL_CMD_MAX,
};
#define REG_GENL_CMD_MAX (__REG_GENL_CMD_MAX - 1)
#define REG_GENL_FAMILY_NAME "reg_event"
#define REG_GENL_VERSION 0x01
#define REG_GENL_MCAST_GROUP_NAME "reg_mc_group"
#endif
