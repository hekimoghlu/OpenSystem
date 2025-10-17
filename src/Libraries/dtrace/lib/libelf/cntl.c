/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 5, 2025.
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
 * Date: Friday, February 3, 2023.
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

int
elf_cntl(Elf * elf, Elf_Cmd cmd)
{

	if (elf == 0)
		return (0);
	ELFWLOCK(elf);
	switch (cmd) {
	case ELF_C_FDREAD:
	{
		int	j = 0;

		if ((elf->ed_myflags & EDF_READ) == 0) {
			_elf_seterr(EREQ_CNTLWRT, 0);
			ELFUNLOCK(elf);
			return (-1);
		}
		if ((elf->ed_status != ES_FROZEN) &&
		    ((_elf_cook(elf) != OK_YES) ||
		    (_elf_vm(elf, (size_t)0, elf->ed_fsz) != OK_YES)))
			j = -1;
		elf->ed_fd = -1;
		ELFUNLOCK(elf);
		return (j);
	}

	case ELF_C_FDDONE:
		if ((elf->ed_myflags & EDF_READ) == 0) {
			_elf_seterr(EREQ_CNTLWRT, 0);
			ELFUNLOCK(elf);
			return (-1);
		}
		elf->ed_fd = -1;
		ELFUNLOCK(elf);
		return (0);

	default:
		_elf_seterr(EREQ_CNTLCMD, 0);
		break;
	}
	ELFUNLOCK(elf);
	return (-1);
}

