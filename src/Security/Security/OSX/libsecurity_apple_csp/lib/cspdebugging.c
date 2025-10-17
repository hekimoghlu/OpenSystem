/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 13, 2023.
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
	File:		cspdebugging.c

	Contains:	Debugging support.


	Copyright (c) 1998,2011,2013-2014 Apple Inc. All Rights Reserved.

	Change History (most recent first): 

		03/10/98	dpm		Created.

*/

#include "cspdebugging.h"

#if		!LOG_VIA_PRINTF

#include <string.h>
#include <TextUtils.h>

/* common log macros */

/* this one needs a writable string */
static void logCom(unsigned char *str) {
	c2pstr((char *)str);
	DebugStr(str);
}

/* remaining ones can take constant strings */
void dblog0(char *str)	{
	Str255	outStr;
	strcpy((char *)outStr, str);
	logCom(outStr);
}

void dblog1(char *str, void *arg1)	{
	Str255	outStr;
	snprintf(outStr, sizeof(outStr), str, arg1);
	logCom(outStr);
}

void dblog2(char *str, void * arg1, void * arg2)	{
	Str255	outStr;
	snprintf(outStr, sizeof(outStr), str, arg1, arg2);
	logCom(outStr);
}

void dblog3(char *str, void * arg1, void * arg2, void * arg3)	{
	Str255	outStr;
	snprintf(outStr, sizeof(outStr), str, arg1, arg2, arg3);
	logCom(outStr);
}

void dblog4(char *str, void * arg1, void * arg2, void * arg3, void * arg4)	{
	Str255	outStr;
	snprintf(outStr, sizeof(outStr), str, arg1, arg2, arg3, arg4);
	logCom(outStr);
}

#endif	/* !LOG_VIA_PRINTF */

//int foobarSymbol;
