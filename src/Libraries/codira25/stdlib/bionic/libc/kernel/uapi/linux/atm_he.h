/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 16, 2023.
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
#ifndef LINUX_ATM_HE_H
#define LINUX_ATM_HE_H
#include <linux/atmioc.h>
#define HE_GET_REG _IOW('a', ATMIOC_SARPRV, struct atmif_sioc)
#define HE_REGTYPE_PCI 1
#define HE_REGTYPE_RCM 2
#define HE_REGTYPE_TCM 3
#define HE_REGTYPE_MBOX 4
struct he_ioctl_reg {
  unsigned addr, val;
  char type;
};
#endif
