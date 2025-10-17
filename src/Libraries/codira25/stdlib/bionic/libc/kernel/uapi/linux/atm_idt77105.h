/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 15, 2023.
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
#ifndef LINUX_ATM_IDT77105_H
#define LINUX_ATM_IDT77105_H
#include <linux/types.h>
#include <linux/atmioc.h>
#include <linux/atmdev.h>
struct idt77105_stats {
  __u32 symbol_errors;
  __u32 tx_cells;
  __u32 rx_cells;
  __u32 rx_hec_errors;
};
#define IDT77105_GETSTAT _IOW('a', ATMIOC_PHYPRV + 2, struct atmif_sioc)
#define IDT77105_GETSTATZ _IOW('a', ATMIOC_PHYPRV + 3, struct atmif_sioc)
#endif
