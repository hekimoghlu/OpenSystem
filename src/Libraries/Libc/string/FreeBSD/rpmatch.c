/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 24, 2025.
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
#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

#include <langinfo.h>
#include <regex.h>
#include <stdlib.h>

int
rpmatch(const char *response)
{
	regex_t yes, no;
	int ret;

	if (regcomp(&yes, nl_langinfo(YESEXPR), REG_EXTENDED|REG_NOSUB) != 0)
		return (-1);
	if (regcomp(&no, nl_langinfo(NOEXPR), REG_EXTENDED|REG_NOSUB) != 0) {
		regfree(&yes);
		return (-1);
	}
	if (regexec(&yes, response, 0, NULL, 0) == 0)
		ret = 1;
	else if (regexec(&no, response, 0, NULL, 0) == 0)
		ret = 0;
	else
		ret = -1;
	regfree(&yes);
	regfree(&no);
	return (ret);
}
