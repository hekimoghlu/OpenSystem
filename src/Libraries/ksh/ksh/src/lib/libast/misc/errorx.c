/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 1, 2024.
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

#include "lclib.h"

/*
 * low level for ERROR_translate()
 * this fills in NiL arg defaults and calls error_info.translate
 */

char*
errorx(const char* loc, const char* cmd, const char* cat, const char* msg)
{
	char*	s;

	if (!error_info.translate)
		error_info.translate = translate; /* 2007-03-19 OLD_Error_info_t workaround */
	if (ERROR_translating())
	{
		if (!loc)
			loc = (const char*)locales[AST_LC_MESSAGES]->code;
		if (!cmd)
			cmd = (const char*)error_info.id;
		if (!cat)
			cat = (const char*)error_info.catalog;
		if (s = (*error_info.translate)(loc, cmd, cat, msg))
			return s;
	}
	return (char*)msg;
}
