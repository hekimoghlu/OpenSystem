/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 5, 2023.
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

#ifndef __APPLE__
__FBSDID("$FreeBSD$");

#ifndef lint
static const char sccsid[] = "@(#)talk.c	8.1 (Berkeley) 6/6/93";
#endif
#endif /* __APPLE__ */

#ifndef lint
__attribute__((__used__))
static const char copyright[] =
"@(#) Copyright (c) 1983, 1993\n\
	The Regents of the University of California.  All rights reserved.\n";
#endif

#include <locale.h>
#include <unistd.h>

#include "talk.h"

/*
 * talk:	A visual form of write. Using sockets, a two way
 *		connection is set up between the two people talking.
 *		With the aid of curses, the screen is split into two
 *		windows, and each users text is added to the window,
 *		one character at a time...
 *
 *		Written by Kipp Hickman
 *
 *		Modified to run under 4.1a by Clem Cole and Peter Moore
 *		Modified to run between hosts by Peter Moore, 8/19/82
 *		Modified to run under 4.1c by Peter Moore 3/17/83
 *		Fixed to not run with unwriteable terminals MRVM 28/12/94
 */

int
main(int argc, char **argv)
{
	(void) setlocale(LC_CTYPE, "");

	get_names(argc, argv);
#ifndef __APPLE__
	setproctitle(" ");
#endif
	check_writeable();
	init_display();
	open_ctl();
	open_sockt();
	start_msgs();
	if (!check_local())
		invite_remote();
	end_msgs();
	set_edit_chars();
	talk();
	return 0;
}
