/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 28, 2023.
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
#ifndef __UAPI_LINUX_PCITEST_H
#define __UAPI_LINUX_PCITEST_H
#define PCITEST_BAR _IO('P', 0x1)
#define PCITEST_INTX_IRQ _IO('P', 0x2)
#define PCITEST_LEGACY_IRQ PCITEST_INTX_IRQ
#define PCITEST_MSI _IOW('P', 0x3, int)
#define PCITEST_WRITE _IOW('P', 0x4, unsigned long)
#define PCITEST_READ _IOW('P', 0x5, unsigned long)
#define PCITEST_COPY _IOW('P', 0x6, unsigned long)
#define PCITEST_MSIX _IOW('P', 0x7, int)
#define PCITEST_SET_IRQTYPE _IOW('P', 0x8, int)
#define PCITEST_GET_IRQTYPE _IO('P', 0x9)
#define PCITEST_CLEAR_IRQ _IO('P', 0x10)
#define PCITEST_FLAGS_USE_DMA 0x00000001
struct pci_endpoint_test_xfer_param {
  unsigned long size;
  unsigned char flags;
};
#endif
