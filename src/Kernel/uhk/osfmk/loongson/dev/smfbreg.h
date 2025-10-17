/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 22, 2024.
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
 * Copyright (c) 2009, 2010 Miodrag Vallat.
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

/*
 * Silicon Motion SM712 registers
 */

/*
 * DPR (2D drawing engine)
 */

#define	DPR_COORDS(x, y)		(((x) << 16) | (y))

#define	SM5XX_DPR_BASE			0x00100000
#define	SM7XX_DPR_BASE			0x00408000
#define	SMXXX_DPR_SIZE			0x00004000

#define	DPR_SRC_COORDS			0x00
#define	DPR_DST_COORDS			0x04
#define	DPR_SPAN_COORDS			0x08
#define	DPR_DE_CTRL			0x0c
#define	DPR_PITCH			0x10
#define	DPR_FG_COLOR			0x14
#define	DPR_BG_COLOR			0x18
#define	DPR_STRETCH			0x1c
#define	DPR_COLOR_COMPARE		0x20
#define	DPR_COLOR_COMPARE_MASK		0x24
#define	DPR_BYTE_BIT_MASK		0x28
#define	DPR_CROP_TOPLEFT_COORDS		0x2c
#define	DPR_CROP_BOTRIGHT_COORDS	0x30
#define	DPR_MONO_PATTERN_LO32		0x34
#define	DPR_MONO_PATTERN_HI32		0x38
#define	DPR_SRC_WINDOW			0x3c
#define	DPR_SRC_BASE			0x40
#define	DPR_DST_BASE			0x44

#define	DE_CTRL_START			0x80000000
#define	DE_CTRL_RTOL			0x08000000
#define	DE_CTRL_COMMAND_MASK		0x001f0000
#define	DE_CTRL_COMMAND_SHIFT			16
#define	DE_CTRL_COMMAND_BITBLT			0x00
#define	DE_CTRL_COMMAND_SOLIDFILL		0x01
#define	DE_CTRL_ROP_ENABLE		0x00008000
#define	DE_CTRL_ROP_MASK		0x000000ff
#define	DE_CTRL_ROP_SHIFT			0
#define	DE_CTRL_ROP_SRC				0x0c

/*
 * VPR (Video Parameter Registers)
 */

#define	SM7XX_VPR_BASE			0x0040c000

/*
 * MMIO (SM7XX only)
 */

#define	SM7XX_MMIO_BASE			0x00700000
#define	SM7XX_MMIO_SIZE			0x00004000
