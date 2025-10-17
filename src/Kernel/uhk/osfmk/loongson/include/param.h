/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 30, 2024.
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
/* Public Domain */

#ifndef	_MACHINE_PARAM_H_
#define	_MACHINE_PARAM_H_

#define	MACHINE		"loongson"
#define	_MACHINE	loongson
#define	MACHINE_ARCH	"mips64el"	/* not the canonical endianness */
#define	_MACHINE_ARCH	mips64el
#define	MACHINE_CPU	"mips64"
#define	_MACHINE_CPU	mips64
#define	MID_MACHINE	MID_MIPS64

#ifdef _KERNEL

/*
 * The Loongson level 1 cache expects software to prevent virtual
 * aliases. Unfortunately, since this cache is physically tagged,
 * this would require all virtual address to have the same bits 14
 * and 13 as their physical addresses, which is not something the
 * kernel can guarantee unless the page size is at least 16KB.
 */
#define	PAGE_SHIFT	14

#endif /* _KERNEL */

#include <mips64/param.h>

#endif /* _MACHINE_PARAM_H_ */
