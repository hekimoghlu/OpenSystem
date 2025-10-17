/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 13, 2024.
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
#ifndef lint
#if 0
static char sccsid[] = "@(#)cmdtab.c	8.1 (Berkeley) 6/6/93";
#endif
#endif /* not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

#include "def.h"
#include "extern.h"

/*
 * Mail -- a mail program
 *
 * Define all of the command names and bindings.
 */

const struct cmd cmdtab[] = {
	/*						msgmask msgflag  */
	/* command	function	argtype		result	& mask   */
	/* -------	--------	-------		------	-------  */
	{ "next",	next,		NDMLIST,	0,	MMNDEL	},
	{ "alias",	group,		M|RAWLIST,	0,	1000 	},
	{ "print",	type,		MSGLIST,	0,	MMNDEL	},
	{ "type",	type,		MSGLIST,	0,	MMNDEL	},
	{ "Type",	Type,		MSGLIST,	0,	MMNDEL	},
	{ "Print",	Type,		MSGLIST,	0,	MMNDEL	},
	{ "visual",	visual,		I|MSGLIST,	0,	MMNORM	},
	{ "top",	top,		MSGLIST,	0,	MMNDEL	},
	{ "touch",	stouch,		W|MSGLIST,	0,	MMNDEL	},
	{ "preserve",	preserve,	W|MSGLIST,	0,	MMNDEL	},
	{ "delete",	deletecmd,	W|P|MSGLIST,	0,	MMNDEL	},
	{ "dp",		deltype,	W|MSGLIST,	0,	MMNDEL	},
	{ "dt",		deltype,	W|MSGLIST,	0,	MMNDEL	},
	{ "undelete",	undeletecmd, 	P|MSGLIST,	MDELETED,MMNDEL },
	{ "unset",	unset,		M|RAWLIST,	1,	1000 	},
	{ "mail",	sendmail,	R|M|I|STRLIST,	0,	0 	},
	{ "mbox",	mboxit,		W|MSGLIST,	0,	0 	},
	{ "more",	more,		MSGLIST,	0,	MMNDEL	},
	{ "page",	more,		MSGLIST,	0,	MMNDEL	},
	{ "More",	More,		MSGLIST,	0,	MMNDEL	},
	{ "Page",	More,		MSGLIST,	0,	MMNDEL	},
	{ "unread",	unread,		MSGLIST,	0,	MMNDEL	},
	{ "!",		shell,		I|STRLIST,	0,	0	},
	{ "copy",	copycmd,	M|STRLIST,	0,	0 	},
	{ "Copy",	Capcopycmd,	MSGLIST,	0,	0 	},
	{ "chdir",	schdir,		M|RAWLIST,	0,	1 	},
	{ "cd",		schdir,		M|RAWLIST,	0,	1 	},
	{ "save",	save,		STRLIST,	0,	0	},
	{ "Save",	Capsave,	STRLIST,	0,	0	},
	{ "source",	source,		M|RAWLIST,	1,	1	},
	{ "set",	set,		M|RAWLIST,	0,	1000 	},
	{ "shell",	dosh,		I|NOLIST,	0,	0 	},
	{ "version",	pversion,	M|NOLIST,	0,	0 	},
	{ "group",	group,		M|RAWLIST,	0,	1000 	},
	{ "write",	swrite,		STRLIST,	0,	0 	},
	{ "from",	from,		MSGLIST,	0,	MMNORM	},
	{ "file",	file,		T|M|RAWLIST,	0,	1 	},
	{ "followup",	followup,	MSGLIST,	0,	0 	},
	{ "folder",	file,		T|M|RAWLIST,	0,	1 	},
	{ "folders",	folders,	T|M|NOLIST,	0,	0 	},
	{ "Followup",	Capfollowup,	MSGLIST,	0,	0 	},
	{ "|",		mailpipe,	STRLIST,	0,	0 	},
	{ "pipe",	mailpipe,	STRLIST,	0,	0 	},
	{ "?",		help,		M|NOLIST,	0,	0 	},
	{ "z",		scroll,		M|STRLIST,	0,	0 	},
	{ "headers",	headers,	MSGLIST,	0,	MMNDEL	},
	{ "help",	help,		M|NOLIST,	0,	0 	},
	{ "=",		pdot,		NOLIST,		0,	0 	},
	{ "Reply",	Respond,	R|I|MSGLIST,	0,	MMNDEL	},
	{ "Respond",	Respond,	R|I|MSGLIST,	0,	MMNDEL	},
	{ "reply",	respond,	R|I|MSGLIST,	0,	MMNDEL	},
	{ "respond",	respond,	R|I|MSGLIST,	0,	MMNDEL	},
	{ "edit",	editor,		I|MSGLIST,	0,	MMNORM	},
	{ "echo",	echo,		M|RAWLIST,	0,	1000 	},
	{ "quit",	quitcmd,	NOLIST,		0,	0 	},
	{ "list",	pcmdlist,	M|NOLIST,	0,	0 	},
	{ "xit",	rexit,		M|NOLIST,	0,	0 	},
	{ "exit",	rexit,		M|NOLIST,	0,	0 	},
	{ "size",	messize,	MSGLIST,	0,	MMNDEL	},
	{ "hold",	preserve,	W|MSGLIST,	0,	MMNDEL	},
	{ "if",		ifcmd,		F|M|RAWLIST,	1,	1 	},
	{ "else",	elsecmd,	F|M|RAWLIST,	0,	0 	},
	{ "endif",	endifcmd,	F|M|RAWLIST,	0,	0 	},
	{ "alternates",	alternates,	M|RAWLIST,	0,	1000 	},
	{ "ignore",	igfield,	M|RAWLIST,	0,	1000 	},
	{ "discard",	igfield,	M|RAWLIST,	0,	1000 	},
	{ "retain",	retfield,	M|RAWLIST,	0,	1000 	},
	{ "saveignore",	saveigfield,	M|RAWLIST,	0,	1000 	},
	{ "savediscard",saveigfield,	M|RAWLIST,	0,	1000 	},
	{ "saveretain",	saveretfield,	M|RAWLIST,	0,	1000 	},
/*	{ "Header",	Header,		STRLIST,	0,	1000	}, */
	{ "core",	core,		M|NOLIST,	0,	0 	},
	{ "#",		null,		M|NOLIST,	0,	0 	},
	{ "clobber",	clobber,	M|RAWLIST,	0,	1 	},
	{ "inc",	inc,		T|NOLIST,	0,	0 	},
	{ 0,		0,		0,		0,	0 	}
};
