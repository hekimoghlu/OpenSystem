/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 26, 2024.
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
 * posix regex fatal error interface to error()
 */

#include "reglib.h"

#include <error.h>

void
regfatalpat(regex_t* p, int level, int code, const char* pat)
{
	char	buf[128];

	regerror(code, p, buf, sizeof(buf));
	regfree(p);
	if (pat)
		error(level, "regular expression: %s: %s", pat, buf);
	else
		error(level, "regular expression: %s", buf);
}

void
regfatal(regex_t* p, int level, int code)
{
	regfatalpat(p, level, code, NiL);
}
