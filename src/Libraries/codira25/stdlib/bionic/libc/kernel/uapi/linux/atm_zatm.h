/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 9, 2024.
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
#ifndef LINUX_ATM_ZATM_H
#define LINUX_ATM_ZATM_H
#include <linux/atmapi.h>
#include <linux/atmioc.h>
#define ZATM_GETPOOL _IOW('a', ATMIOC_SARPRV + 1, struct atmif_sioc)
#define ZATM_GETPOOLZ _IOW('a', ATMIOC_SARPRV + 2, struct atmif_sioc)
#define ZATM_SETPOOL _IOW('a', ATMIOC_SARPRV + 3, struct atmif_sioc)
struct zatm_pool_info {
  int ref_count;
  int low_water, high_water;
  int rqa_count, rqu_count;
  int offset, next_off;
  int next_cnt, next_thres;
};
struct zatm_pool_req {
  int pool_num;
  struct zatm_pool_info info;
};
#define ZATM_OAM_POOL 0
#define ZATM_AAL0_POOL 1
#define ZATM_AAL5_POOL_BASE 2
#define ZATM_LAST_POOL ZATM_AAL5_POOL_BASE + 10
#define ZATM_TIMER_HISTORY_SIZE 16
#endif
