/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 20, 2021.
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
 * Date: Wednesday, December 18, 2024.
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

#include "libelf.h"
#include "decl.h"
#include "msg.h"


Elf_Scn *
elf_getscn(Elf * elf, size_t index)
{
	Elf_Scn	*	s;
	Elf_Scn	*	prev_s;
	size_t		j = index;
	size_t		tabsz;

	if (elf == 0)
		return (0);

	ELFRLOCK(elf)
	tabsz = elf->ed_scntabsz;
	if (elf->ed_hdscn == 0) {
		ELFUNLOCK(elf)
		ELFWLOCK(elf)
		if ((elf->ed_hdscn == 0) && (_elf_cook(elf) != OK_YES)) {
			ELFUNLOCK(elf);
			return (0);
		}
		ELFUNLOCK(elf);
		ELFRLOCK(elf)
	}
	/*
	 * If the section in question is part of a table allocated
	 * from within _elf_prescn() then we can index straight
	 * to it.
	 */
	if (index < tabsz) {
		s = &elf->ed_hdscn[index];
		ELFUNLOCK(elf);
		return (s);
	}

	if (tabsz)
		s = &elf->ed_hdscn[tabsz - 1];
	else
		s = elf->ed_hdscn;

	for (prev_s = 0; s != 0; prev_s = s, s = s->s_next) {
		if (prev_s) {
			SCNUNLOCK(prev_s)
		}
		SCNLOCK(s)
		if (j == 0) {
			if (s->s_index == index) {
				SCNUNLOCK(s)
				ELFUNLOCK(elf);
				return (s);
			}
			_elf_seterr(EBUG_SCNLIST, 0);
			SCNUNLOCK(s)
			ELFUNLOCK(elf)
			return (0);
		}
		--j;
	}
	if (prev_s) {
		SCNUNLOCK(prev_s)
	}
	_elf_seterr(EREQ_NDX, 0);
	ELFUNLOCK(elf);
	return (0);
}

