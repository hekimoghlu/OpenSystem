/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 27, 2023.
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
 * __os_open --
 *	Open a file descriptor (including page size and log size information).
 */
int
__os_open(env, name, page_size, flags, mode, fhpp)
	ENV *env;
	const char *name;
	u_int32_t page_size, flags;
	int mode;
	DB_FH **fhpp;
{
	OpenFileMode oflags;
	int ret;

	COMPQUIET(page_size, 0);

#define	OKFLAGS								\
	(DB_OSO_ABSMODE | DB_OSO_CREATE | DB_OSO_DIRECT | DB_OSO_DSYNC |\
	DB_OSO_EXCL | DB_OSO_RDONLY | DB_OSO_REGION |	DB_OSO_SEQ |	\
	DB_OSO_TEMP | DB_OSO_TRUNC)
	if ((ret = __db_fchk(env, "__os_open", flags, OKFLAGS)) != 0)
		return (ret);

	oflags = 0;
	if (LF_ISSET(DB_OSO_CREATE))
		oflags |= _OFM_CREATE;

	if (LF_ISSET(DB_OSO_RDONLY))
		oflags |= _OFM_READ;
	else
		oflags |= _OFM_READWRITE;

	if ((ret = __os_openhandle(env, name, oflags, mode, fhpp)) != 0)
		return (ret);
	if (LF_ISSET(DB_OSO_REGION))
		F_SET(fhp, DB_FH_REGION);
	return (ret);
}
