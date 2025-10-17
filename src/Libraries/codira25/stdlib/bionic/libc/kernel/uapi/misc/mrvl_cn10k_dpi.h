/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 1, 2022.
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
#ifndef __MRVL_CN10K_DPI_H__
#define __MRVL_CN10K_DPI_H__
#include <linux/types.h>
#define DPI_MAX_ENGINES 6
struct dpi_mps_mrrs_cfg {
  __u16 max_read_req_sz;
  __u16 max_payload_sz;
  __u16 port;
  __u16 reserved;
};
struct dpi_engine_cfg {
  __u64 fifo_mask;
  __u16 molr[DPI_MAX_ENGINES];
  __u16 update_molr;
  __u16 reserved;
};
#define DPI_MAGIC_NUM 0xB8
#define DPI_MPS_MRRS_CFG _IOW(DPI_MAGIC_NUM, 1, struct dpi_mps_mrrs_cfg)
#define DPI_ENGINE_CFG _IOW(DPI_MAGIC_NUM, 2, struct dpi_engine_cfg)
#endif
