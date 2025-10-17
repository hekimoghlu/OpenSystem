/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 2, 2024.
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
#ifndef LINUX_ATM_NICSTAR_H
#define LINUX_ATM_NICSTAR_H
#include <linux/atmapi.h>
#include <linux/atmioc.h>
#define NS_GETPSTAT _IOWR('a', ATMIOC_SARPRV + 1, struct atmif_sioc)
#define NS_SETBUFLEV _IOW('a', ATMIOC_SARPRV + 2, struct atmif_sioc)
#define NS_ADJBUFLEV _IO('a', ATMIOC_SARPRV + 3)
typedef struct buf_nr {
  unsigned min;
  unsigned init;
  unsigned max;
} buf_nr;
typedef struct pool_levels {
  int buftype;
  int count;
  buf_nr level;
} pool_levels;
#define NS_BUFTYPE_SMALL 1
#define NS_BUFTYPE_LARGE 2
#define NS_BUFTYPE_HUGE 3
#define NS_BUFTYPE_IOVEC 4
#endif
