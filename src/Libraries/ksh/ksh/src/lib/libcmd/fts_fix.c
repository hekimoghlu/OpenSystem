/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 17, 2023.
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
#pragma prototyped
/*
 * -lcmd specific workaround to handle
 *	fts_namelen
 *	fts_pathlen
 *	fts_level
 * changing from [unsigned] short bit to [s]size_t
 *
 * ksh (or any other main application) that pulls in -lcmd
 * at runtime may result in old -last running with new -lcmd
 * which is not a good situation (tm)
 *
 * probably safe to drop after 20150101
 */

#include <ast.h>
#include <fts_fix.h>

#undef	fts_read

FTSENT*
_fts_read(FTS* fts)
{
	FTSENT*		oe;

	static FTSENT*	ne;

	if ((oe = _ast_fts_read(fts)) && ast.version < 20100102L && (ne || (ne = newof(0, FTSENT, 1, 0))))
	{
		*ne = *oe;
		oe = ne;
		ne->fts_namelen = ne->_fts_namelen;
		ne->fts_pathlen = ne->_fts_pathlen;
		ne->fts_level = ne->_fts_level;
	}
	return oe;
}
