/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 5, 2024.
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
#ifndef __PSP_DBC_USER_H__
#define __PSP_DBC_USER_H__
#include <linux/types.h>
#define DBC_NONCE_SIZE 16
#define DBC_SIG_SIZE 32
#define DBC_UID_SIZE 16
struct dbc_user_nonce {
  __u32 auth_needed;
  __u8 nonce[DBC_NONCE_SIZE];
  __u8 signature[DBC_SIG_SIZE];
} __attribute__((__packed__));
struct dbc_user_setuid {
  __u8 uid[DBC_UID_SIZE];
  __u8 signature[DBC_SIG_SIZE];
} __attribute__((__packed__));
struct dbc_user_param {
  __u32 msg_index;
  __u32 param;
  __u8 signature[DBC_SIG_SIZE];
} __attribute__((__packed__));
#define DBC_IOC_TYPE 'D'
#define DBCIOCNONCE _IOWR(DBC_IOC_TYPE, 0x1, struct dbc_user_nonce)
#define DBCIOCUID _IOW(DBC_IOC_TYPE, 0x2, struct dbc_user_setuid)
#define DBCIOCPARAM _IOWR(DBC_IOC_TYPE, 0x3, struct dbc_user_param)
enum dbc_cmd_msg {
  PARAM_GET_FMAX_CAP = 0x3,
  PARAM_SET_FMAX_CAP = 0x4,
  PARAM_GET_PWR_CAP = 0x5,
  PARAM_SET_PWR_CAP = 0x6,
  PARAM_GET_GFX_MODE = 0x7,
  PARAM_SET_GFX_MODE = 0x8,
  PARAM_GET_CURR_TEMP = 0x9,
  PARAM_GET_FMAX_MAX = 0xA,
  PARAM_GET_FMAX_MIN = 0xB,
  PARAM_GET_SOC_PWR_MAX = 0xC,
  PARAM_GET_SOC_PWR_MIN = 0xD,
  PARAM_GET_SOC_PWR_CUR = 0xE,
};
#endif
