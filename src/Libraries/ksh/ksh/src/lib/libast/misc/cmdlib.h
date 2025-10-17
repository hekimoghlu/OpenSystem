/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 30, 2021.
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
 * cmdarg library private definitions
 */

#ifndef _CMDLIB_H
#define _CMDLIB_H	1

#define _CMDARG_PRIVATE_ \
	struct \
	{ \
	size_t		args;		/* total args			*/ \
	size_t		commands;	/* total commands		*/ \
	}		total; \
	Error_f		errorf;		/* optional error callback	*/ \
	Cmdrun_f	runf;		/* exec function		*/ \
	int		argcount;	/* current arg count		*/ \
	int		argmax;		/* max # args			*/ \
	int		echo;		/* just an echo			*/ \
	int		flags;		/* CMD_* flags			*/ \
	int		insertlen;	/* strlen(insert)		*/ \
	int		offset;		/* post arg offset		*/ \
	Cmddisc_t*	disc;		/* discipline			*/ \
	char**		argv;		/* exec argv			*/ \
	char**		firstarg;	/* first argv file arg		*/ \
	char**		insertarg;	/* argv before insert		*/ \
	char**		postarg;	/* start of post arg list	*/ \
	char**		nextarg;	/* next argv file arg		*/ \
	char*		nextstr;	/* next string ends before here	*/ \
	char*		laststr;	/* last string ends before here	*/ \
	char*		insert;		/* replace with current arg	*/ \
	char		buf[1];		/* argv and arg buffer		*/

#include <cmdarg.h>

#endif
