/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 14, 2023.
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
 * Copyright 2008 Sun Microsystems, Inc.  All rights reserved.
 * Use is subject to license terms.
 */

#ifndef	_SYS_MACHELF_H
#define	_SYS_MACHELF_H

#ifdef	__cplusplus
extern "C" {
#endif
#include <sys/elf.h>

/*
 * Make machine class dependent data types transparent to the common code
 */
#if defined(_ELF64) && !defined(_ELF32_COMPAT)
typedef	Elf64_Ehdr	Ehdr;
typedef	Elf64_Shdr	Shdr;
#else	/* _ILP32 */
typedef	Elf32_Ehdr	Ehdr;
typedef	Elf32_Shdr	Shdr;
#endif	/* _ILP32 */

#ifdef	__cplusplus
}
#endif

#endif	/* _SYS_MACHELF_H */
