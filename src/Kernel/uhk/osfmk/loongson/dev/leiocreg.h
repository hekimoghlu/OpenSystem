/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 25, 2022.
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
/*
 * Copyright (c) 2016 Visa Hankala
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#ifndef _LOONGSON_DEV_LEIOCREG_H_
#define _LOONGSON_DEV_LEIOCREG_H_

#define LEIOC_PCI_CFG_BASE	0x1fe00000
#define LEIOC_IO_CFG_BASE	0x1fe00100
#define LEIOC_UART0_BASE	0x1fe001e0
#define LEIOC_UART1_BASE	0x1fe001e8
#define LEIOC_SPI_BASE		0x1fe001f0

#define LEIOC_LPC_MEM_BASE	0x1c000000
#define LEIOC_LPC_MEM_SIZE	0x01000000
#define LEIOC_LPC_BOOT_BASE	0x1fc00000
#define LEIOC_LPC_BOOT_SIZE	0x00100000
#define LEIOC_LPC_REG_BASE	0x1fe00200
#define LEIOC_LPC_REG_SIZE	0x00000100
#define LEIOC_LPC_IO_BASE	0x1ff00000
#define LEIOC_LPC_IO_SIZE	0x00010000

#endif /* _LOONGSON_DEV_LEIOCREG_H_ */
