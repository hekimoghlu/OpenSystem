/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 26, 2025.
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
if (nbytes <= sizeof(elfhdr))
		return 0;

	u.l = 1;
	(void)memcpy(&elfhdr, buf, sizeof elfhdr);
	swap = (u.c[sizeof(int32_t) - 1] + 1) != elfhdr.e_ident[EI_DATA];

	type = elf_getu16(swap, elfhdr.e_type);
	notecount = ms->elf_notes_max;
	switch (type) {
#ifdef ELFCORE
	case ET_CORE:
		phnum = elf_getu16(swap, elfhdr.e_phnum);
		if (phnum > ms->elf_phnum_max)
			return toomany(ms, "program headers", phnum);
		flags |= FLAGS_IS_CORE;
		if (dophn_core(ms, clazz, swap, fd,
		    CAST(off_t, elf_getu(swap, elfhdr.e_phoff)), phnum,
		    CAST(size_t, elf_getu16(swap, elfhdr.e_phentsize)),
		    fsize, &flags, &notecount) == -1)
			return -1;
		break;
#endif
	case ET_EXEC:
	case ET_DYN:
		phnum = elf_getu16(swap, elfhdr.e_phnum);
		if (phnum > ms->elf_phnum_max)
			return toomany(ms, "program", phnum);
		shnum = elf_getu16(swap, elfhdr.e_shnum);
		if (shnum > ms->elf_shnum_max)
			return toomany(ms, "section", shnum);
		if (dophn_exec(ms, clazz, swap, fd,
		    CAST(off_t, elf_getu(swap, elfhdr.e_phoff)), phnum,
		    CAST(size_t, elf_getu16(swap, elfhdr.e_phentsize)),
		    fsize, shnum, &flags, &notecount) == -1)
			return -1;
		/*FALLTHROUGH*/
	case ET_REL:
		shnum = elf_getu16(swap, elfhdr.e_shnum);
		if (shnum > ms->elf_shnum_max)
			return toomany(ms, "section headers", shnum);
		if (doshn(ms, clazz, swap, fd,
		    CAST(off_t, elf_getu(swap, elfhdr.e_shoff)), shnum,
		    CAST(size_t, elf_getu16(swap, elfhdr.e_shentsize)),
		    fsize, elf_getu16(swap, elfhdr.e_machine),
		    CAST(int, elf_getu16(swap, elfhdr.e_shstrndx)),
		    &flags, &notecount) == -1)
			return -1;
		break;

	default:
		break;
	}
	if (notecount == 0)
		return toomany(ms, "notes", ms->elf_notes_max);
	return 1;
