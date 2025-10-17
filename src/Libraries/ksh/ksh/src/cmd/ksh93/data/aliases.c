/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 21, 2023.
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
#include	"defs.h"
#include	<signal.h>
#include	"FEATURE/options"
#include	"FEATURE/dynamic"

/*
 * This is the table of built-in aliases.  These should be exported.
 */

const struct shtable2 shtab_aliases[] =
{
#if SHOPT_FS_3D
	"2d",		NV_NOFREE,		"set -f;_2d",
#endif /* SHOPT_FS_3D */
	"autoload",	NV_NOFREE,		"typeset -fu",
	"command",	NV_NOFREE,		"command ",
	"compound",	NV_NOFREE|BLT_DCL,	"typeset -C",
	"fc",		NV_NOFREE,		"hist",
	"float",	NV_NOFREE|BLT_DCL,	"typeset -lE",
	"functions",	NV_NOFREE,		"typeset -f",
	"hash",		NV_NOFREE,		"alias -t --",
	"history",	NV_NOFREE,		"hist -l",
	"integer",	NV_NOFREE|BLT_DCL,	"typeset -li",
	"nameref",	NV_NOFREE|BLT_DCL,	"typeset -n",
	"nohup",	NV_NOFREE,		"nohup ",
	"r",		NV_NOFREE,		"hist -s",
	"redirect",	NV_NOFREE,		"command exec",
	"source",	NV_NOFREE,		"command .",
#ifdef SIGTSTP
	"stop",		NV_NOFREE,		"kill -s STOP",
	"suspend", 	NV_NOFREE,		"kill -s STOP $$",
#endif /*SIGTSTP */
	"times",	NV_NOFREE,		"{ { time;} 2>&1;}",
	"type",		NV_NOFREE,		"whence -v",
	"",		0,			(char*)0
};

