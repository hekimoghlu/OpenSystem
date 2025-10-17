/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 14, 2022.
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
 * Copyright 2002 Sun Microsystems, Inc.  All rights reserved.
 * Use is subject to license terms.
 */

#include <string.h>
#include <gelf.h>
#include <decl.h>
#include <msg.h>

int
elf_getshstrndx(Elf *elf, size_t *shstrndx)
{
	GElf_Ehdr	ehdr;
	Elf_Scn		*scn;
	GElf_Shdr	shdr0;

	if (gelf_getehdr(elf, &ehdr) == 0)
		return (0);
	if (ehdr.e_shstrndx != SHN_XINDEX) {
		*shstrndx = ehdr.e_shstrndx;
		return (1);
	}
	if ((scn = elf_getscn(elf, 0)) == 0)
		return (0);
	if (gelf_getshdr(scn, &shdr0) == 0)
		return (0);
	*shstrndx = shdr0.sh_link;
	return (1);
}
