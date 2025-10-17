/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 2, 2023.
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
 * Date: Friday, November 1, 2024.
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


char *
elf_getident(Elf * elf, size_t * ptr)
{
	size_t	sz = 0;
	char *	id = 0;

	if (elf != 0) {
		ELFRLOCK(elf)
		if (elf->ed_identsz != 0) {
			if ((elf->ed_vm == 0) || (elf->ed_status !=
			    ES_COOKED)) {
				/*
				 * We need to upgrade to a Writers
				 * lock
				 */
				ELFUNLOCK(elf)
				ELFWLOCK(elf)
				if ((_elf_cook(elf) == OK_YES) &&
				    (_elf_vm(elf, (size_t)0,
				    elf->ed_identsz) == OK_YES)) {
					id = elf->ed_ident;
					sz = elf->ed_identsz;
				}
			} else {
				id = elf->ed_ident;
				sz = elf->ed_identsz;
			}
		}
		ELFUNLOCK(elf)
	}
	if (ptr != 0)
		*ptr = sz;
	return (id);
}

char *
elf_getimage(Elf * elf, size_t * ptr)
{
	char *image = NULL;

	ELFRLOCK(elf)

	if (ptr) {
		*ptr = elf->ed_imagesz;
	}
	image = elf->ed_image;

	ELFUNLOCK(elf)

	return image;
}

