/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 8, 2025.
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
#ifndef	_FINGER_H_
#define	_FINGER_H_

typedef struct person {
	uid_t uid;			/* user id */
	char *dir;			/* user's home directory */
	char *homephone;		/* pointer to home phone no. */
	char *name;			/* login name */
	char *office;			/* pointer to office name */
	char *officephone;		/* pointer to office phone no. */
	char *realname;			/* pointer to full name */
	char *shell;			/* user's shell */
	time_t mailread;		/* last time mail was read */
	time_t mailrecv;		/* last time mail was received */
	struct where *whead, *wtail;	/* list of where user is or has been */
} PERSON;

enum status { LASTLOG, LOGGEDIN };

typedef struct where {
	struct where *next;		/* next place user is or has been */
	enum status info;		/* type/status of request */
	short writable;			/* tty is writable */
	time_t loginat;			/* time of (last) login */
	time_t idletime;		/* how long idle (if logged in) */
	char tty[sizeof ((struct utmpx *)0)->ut_line];  /* tty line */
	char host[sizeof ((struct utmpx *)0)->ut_host]; /* host name */
} WHERE;

#define UNPRIV_NAME	"nobody"	/* Preferred privilege level */
#define UNPRIV_UGID	32767		/* Default uid and gid */
#define OUTPUT_MAX	100000		/* Do not keep listinging forever */
#define TIME_LIMIT	360		/* Do not keep listinging forever */

#include "extern.h"

#endif /* !_FINGER_H_ */
