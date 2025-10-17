/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 8, 2022.
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

#include <ast.h>

#undef	_lib_getopt	/* we can satisfy the api */

#if _lib_getopt

NoN(getopt)

#else

#undef	_BLD_ast	/* enable ast imports since we're user static */

#include <error.h>
#include <option.h>

int		opterr = 1;
int		optind = 1;
int		optopt = 0;
char*		optarg = 0;

static int	lastoptind;

extern int
getopt(int argc, char* const* argv, const char* optstring)
{
	int	n;

	NoP(argc);
	opt_info.index = (optind > 1 || optind == lastoptind) ? optind : 0;
	if (opt_info.index >= argc)
		return -1;
	switch (n = optget((char**)argv, optstring))
	{
	case ':':
		n = '?';
		/*FALLTHROUGH*/
	case '?':
		if (opterr && (!optstring || *optstring != ':'))
		{
			if (!error_info.id)
				error_info.id = argv[0];
			errormsg(NiL, 2, opt_info.arg);
		}
		optopt = opt_info.option[1];
		break;
	case 0:
		n = -1;
		break;
	}
	optarg = opt_info.arg;
	lastoptind = optind = opt_info.index;
	return n;
}

#endif
