/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 9, 2024.
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
#ifndef _UAPI__HPET__
#define _UAPI__HPET__
#include <linux/compiler.h>
struct hpet_info {
  unsigned long hi_ireqfreq;
  unsigned long hi_flags;
  unsigned short hi_hpet;
  unsigned short hi_timer;
};
#define HPET_INFO_PERIODIC 0x0010
#define HPET_IE_ON _IO('h', 0x01)
#define HPET_IE_OFF _IO('h', 0x02)
#define HPET_INFO _IOR('h', 0x03, struct hpet_info)
#define HPET_EPI _IO('h', 0x04)
#define HPET_DPI _IO('h', 0x05)
#define HPET_IRQFREQ _IOW('h', 0x6, unsigned long)
#define MAX_HPET_TBS 8
#endif
