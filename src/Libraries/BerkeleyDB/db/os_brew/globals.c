/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 24, 2023.
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
#include "db_config.h"

#include "db_int.h"

/*
 * brew_bdb_begin --
 *	Initialize the BREW port of Berkeley DB.
 */
int
brew_bdb_begin()
{
	void *p;

	/*
	 * The BREW ARM compiler can't handle statics or globals, so we have
	 * store them off the AEEApplet and initialize them in in-line code.
	 */
	p = ((BDBApp *)GETAPPINSTANCE())->db_global_values;
	if (p == NULL) {
		if ((p = malloc(sizeof(DB_GLOBALS))) == NULL)
			return (ENOMEM);
		memset(p, 0, sizeof(DB_GLOBALS));

		((BDBApp *)GETAPPINSTANCE())->db_global_values = p;

		TAILQ_INIT(&DB_GLOBAL(envq));
		DB_GLOBAL(db_line) =
		    "=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=";
	}
	return (0);
}

/*
 * brew_bdb_end --
 *	Close down the BREW port of Berkeley DB.
 */
void
brew_bdb_end()
{
	void *p;

	p = ((BDBApp *)GETAPPINSTANCE())->db_global_values;

	free(p);
}
