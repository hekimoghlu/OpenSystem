/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 22, 2022.
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
#ifndef _FC_GS_H_
#define _FC_GS_H_
#include <linux/types.h>
struct fc_ct_hdr {
  __u8 ct_rev;
  __u8 ct_in_id[3];
  __u8 ct_fs_type;
  __u8 ct_fs_subtype;
  __u8 ct_options;
  __u8 _ct_resvd1;
  __be16 ct_cmd;
  __be16 ct_mr_size;
  __u8 _ct_resvd2;
  __u8 ct_reason;
  __u8 ct_explan;
  __u8 ct_vendor;
};
#define FC_CT_HDR_LEN 16
enum fc_ct_rev {
  FC_CT_REV = 1
};
enum fc_ct_fs_type {
  FC_FST_ALIAS = 0xf8,
  FC_FST_MGMT = 0xfa,
  FC_FST_TIME = 0xfb,
  FC_FST_DIR = 0xfc,
};
enum fc_ct_cmd {
  FC_FS_RJT = 0x8001,
  FC_FS_ACC = 0x8002,
};
enum fc_ct_reason {
  FC_FS_RJT_CMD = 0x01,
  FC_FS_RJT_VER = 0x02,
  FC_FS_RJT_LOG = 0x03,
  FC_FS_RJT_IUSIZ = 0x04,
  FC_FS_RJT_BSY = 0x05,
  FC_FS_RJT_PROTO = 0x07,
  FC_FS_RJT_UNABL = 0x09,
  FC_FS_RJT_UNSUP = 0x0b,
};
enum fc_ct_explan {
  FC_FS_EXP_NONE = 0x00,
  FC_FS_EXP_PID = 0x01,
  FC_FS_EXP_PNAM = 0x02,
  FC_FS_EXP_NNAM = 0x03,
  FC_FS_EXP_COS = 0x04,
  FC_FS_EXP_FTNR = 0x07,
};
#endif
