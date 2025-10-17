/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 19, 2025.
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
#ifndef __HSI_CHAR_H
#define __HSI_CHAR_H
#include <linux/types.h>
#define HSI_CHAR_MAGIC 'k'
#define HSC_IOW(num,dtype) _IOW(HSI_CHAR_MAGIC, num, dtype)
#define HSC_IOR(num,dtype) _IOR(HSI_CHAR_MAGIC, num, dtype)
#define HSC_IOWR(num,dtype) _IOWR(HSI_CHAR_MAGIC, num, dtype)
#define HSC_IO(num) _IO(HSI_CHAR_MAGIC, num)
#define HSC_RESET HSC_IO(16)
#define HSC_SET_PM HSC_IO(17)
#define HSC_SEND_BREAK HSC_IO(18)
#define HSC_SET_RX HSC_IOW(19, struct hsc_rx_config)
#define HSC_GET_RX HSC_IOW(20, struct hsc_rx_config)
#define HSC_SET_TX HSC_IOW(21, struct hsc_tx_config)
#define HSC_GET_TX HSC_IOW(22, struct hsc_tx_config)
#define HSC_PM_DISABLE 0
#define HSC_PM_ENABLE 1
#define HSC_MODE_STREAM 1
#define HSC_MODE_FRAME 2
#define HSC_FLOW_SYNC 0
#define HSC_ARB_RR 0
#define HSC_ARB_PRIO 1
struct hsc_rx_config {
  __u32 mode;
  __u32 flow;
  __u32 channels;
};
struct hsc_tx_config {
  __u32 mode;
  __u32 channels;
  __u32 speed;
  __u32 arb_mode;
};
#endif
