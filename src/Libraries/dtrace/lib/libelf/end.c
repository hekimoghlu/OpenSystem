/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 15, 2023.
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
/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 27, 2022.
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
 *//*	  All Rights Reserved  	*/

#include <ar.h>
#include <stdlib.h>
#include "libelf.h"
#include "decl.h"

int
elf_end(Elf * elf)
{
	Elf_Scn *	s;
	Dnode *	d;
	Elf_Void *		trail = 0;
	int			rc;

	if (elf == 0)
		return (0);

	ELFWLOCK(elf)
	if (--elf->ed_activ != 0) {
		rc = elf->ed_activ;
		ELFUNLOCK(elf)
		return (rc);
	}

	while (elf->ed_activ == 0) {
		for (s = elf->ed_hdscn; s != 0; s = s->s_next) {
			if (s->s_myflags & SF_ALLOC) {
				if (trail != 0)
					free(trail);
				trail = (Elf_Void *)s;
			}

			if ((s->s_myflags & SF_READY) == 0)
				continue;
			for (d = s->s_hdnode; d != 0; ) {
				register Dnode	*t;

				if (d->db_buf != 0)
					free(d->db_buf);
				if ((t = d->db_raw) != 0) {
					if (t->db_buf != 0)
						free(t->db_buf);
					if (t->db_myflags & DBF_ALLOC)
						free(t);
				}
				t = d->db_next;
				if (d->db_myflags & DBF_ALLOC)
					free(d);
				d = t;
			}
		}
		if (trail != 0) {
			free(trail);
			trail = 0;
		}

		if (elf->ed_kind == ELF_K_MACHO) {
			free(elf->ed_ident);
		}
		if (elf->ed_myflags & EDF_EHALLOC)
			free(elf->ed_ehdr);
		if (elf->ed_myflags & EDF_PHALLOC)
			free(elf->ed_phdr);
		if (elf->ed_myflags & EDF_SHALLOC)
			free(elf->ed_shdr);

		/*
		 * Don't release the image until the last reference dies.
		 * If the image was introduced via elf_memory() then
		 * we don't release it at all, it's not ours to release.
		 */

		if (elf->ed_parent == 0) {
			if (elf->ed_vm != 0)
				free(elf->ed_vm);
		}
		trail = (Elf_Void *)elf;
		elf = elf->ed_parent;
		ELFUNLOCK(trail)
		free(trail);
		if (elf == 0)
			break;
		/*
		 * If parent is inactive we close
		 * it too, so we need to lock it too.
		 */
		ELFWLOCK(elf)
		--elf->ed_activ;
	}

	if (elf) {
		ELFUNLOCK(elf)
	}

	return (0);
}

