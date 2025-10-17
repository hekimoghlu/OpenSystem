/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 26, 2022.
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
#ifndef _M88K_EXEC_H_
#define _M88K_EXEC_H_

#define __LDPGSZ        4096

#define ARCH_ELFSIZE		32

#define ELF_TARG_CLASS		ELFCLASS32
#define ELF_TARG_DATA		ELFDATA2MSB
#define ELF_TARG_MACH		EM_88K

/* Processor specific dynamic tag values.  */
#define	DT_88K_ADDRBASE	0x70000001
#define	DT_88K_PLTSTART	0x70000002
#define	DT_88K_PLTEND	0x70000003
#define	DT_88K_TDESC	0x70000004

#define	DT_PROCNUM	(DT_88K_TDESC + 1 - DT_LOPROC)

#endif /* _M88K_EXEC_H_ */
