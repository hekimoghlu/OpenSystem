/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 4, 2024.
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
#ifndef _UAPILINUX_PCI_H
#define _UAPILINUX_PCI_H
#include <linux/pci_regs.h>
#define PCI_DEVFN(slot,func) ((((slot) & 0x1f) << 3) | ((func) & 0x07))
#define PCI_SLOT(devfn) (((devfn) >> 3) & 0x1f)
#define PCI_FUNC(devfn) ((devfn) & 0x07)
#define PCIIOC_BASE ('P' << 24 | 'C' << 16 | 'I' << 8)
#define PCIIOC_CONTROLLER (PCIIOC_BASE | 0x00)
#define PCIIOC_MMAP_IS_IO (PCIIOC_BASE | 0x01)
#define PCIIOC_MMAP_IS_MEM (PCIIOC_BASE | 0x02)
#define PCIIOC_WRITE_COMBINE (PCIIOC_BASE | 0x03)
#endif
