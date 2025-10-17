/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 26, 2023.
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
#ifndef _UAPI__DLM_DOT_H__
#define _UAPI__DLM_DOT_H__
#include <linux/dlmconstants.h>
#include <linux/types.h>
typedef void dlm_lockspace_t;
#define DLM_SBF_DEMOTED 0x01
#define DLM_SBF_VALNOTVALID 0x02
#define DLM_SBF_ALTMODE 0x04
struct dlm_lksb {
  int sb_status;
  __u32 sb_lkid;
  char sb_flags;
  char * sb_lvbptr;
};
#define DLM_LSFL_TIMEWARN 0x00000002
#define DLM_LSFL_NEWEXCL 0x00000008
#define __DLM_LSFL_RESERVED0 0x00000010
#endif
