/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 7, 2023.
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
#ifndef _UAPI_FSL_MC_H_
#define _UAPI_FSL_MC_H_
#include <linux/types.h>
#define MC_CMD_NUM_OF_PARAMS 7
struct fsl_mc_command {
  __le64 header;
  __le64 params[MC_CMD_NUM_OF_PARAMS];
};
#define FSL_MC_SEND_CMD_IOCTL_TYPE 'R'
#define FSL_MC_SEND_CMD_IOCTL_SEQ 0xE0
#define FSL_MC_SEND_MC_COMMAND _IOWR(FSL_MC_SEND_CMD_IOCTL_TYPE, FSL_MC_SEND_CMD_IOCTL_SEQ, struct fsl_mc_command)
#endif
