/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 8, 2022.
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
 * openlog implementation
 */

#include <ast.h>

#if _lib_syslog

NoN(openlog)

#else

#include "sysloglib.h"

void
openlog(const char* ident, int flags, int facility)
{
	int		n;

	if (ident)
	{
		n = strlen(ident);
		if (n >= sizeof(log.ident))
			n = sizeof(log.ident) - 1;
		memcpy(log.ident, ident, n);
		log.ident[n] = 0;
	}
	else
		log.ident[0] = 0;
	log.facility = facility;
	log.flags = flags;
	if (!(log.flags & LOG_ODELAY))
		sendlog(NiL);
}

#endif
