/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 26, 2023.
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
/*	$NetBSD: vmparam.h,v 1.1 2006/09/01 21:26:18 uwe Exp $	*/

#ifndef _MACHINE_VMPARAM_H_
#define _MACHINE_VMPARAM_H_

#include <sh/vmparam.h>

#define	KERNBASE		0x8c000000

#define VM_PHYSSEG_MAX		1
#define	VM_PHYSSEG_NOADD
#define	VM_PHYSSEG_STRAT	VM_PSTRAT_RANDOM

#endif /* _MACHINE_VMPARAM_H_ */
