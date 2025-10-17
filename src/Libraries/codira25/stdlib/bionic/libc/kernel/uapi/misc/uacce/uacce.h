/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 19, 2023.
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
#ifndef _UAPIUUACCE_H
#define _UAPIUUACCE_H
#include <linux/types.h>
#include <linux/ioctl.h>
#define UACCE_CMD_START_Q _IO('W', 0)
#define UACCE_CMD_PUT_Q _IO('W', 1)
#define UACCE_DEV_SVA BIT(0)
enum uacce_qfrt {
  UACCE_QFRT_MMIO = 0,
  UACCE_QFRT_DUS = 1,
};
#endif
