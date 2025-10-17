/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 4, 2025.
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
 * Copyright 2004 Sun Microsystems, Inc.  All rights reserved.
 * Use is subject to license terms.
 */

#include <sys/types.h>
#include <string.h>

#include "symbol.h"

int
ignore_symbol(GElf_Sym *sym, const char *name)
{
	uchar_t type = GELF_ST_TYPE(sym->st_info);

	/*
	 * As an optimization, we do not output function or data object
	 * records for undefined or anonymous symbols.
	 */
	if (sym->st_shndx == SHN_UNDEF || sym->st_name == 0)
		return (1);

	/*
	 * _START_ and _END_ are added to the symbol table by the
	 * linker, and will never have associated type information.
	 */
	if (strcmp(name, "_START_") == 0 || strcmp(name, "_END_") == 0)
		return (1);

	/*
	 * Do not output records for absolute-valued object symbols
	 * that have value zero.  The compiler insists on generating
	 * things like this for __fsr_init_value settings, etc.
	 */
	if (type == STT_OBJECT && sym->st_shndx == SHN_ABS &&
	    sym->st_value == 0)
		return (1);
	return (0);
}
