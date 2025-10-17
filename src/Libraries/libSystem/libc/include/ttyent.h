/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 17, 2023.
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
#ifndef	_TTYENT_H_
#define	_TTYENT_H_

#define	_PATH_TTYS	"/etc/ttys"

#define	_TTYS_OFF	"off"
#define	_TTYS_ON	"on"
#define	_TTYS_SECURE	"secure"
#define	_TTYS_WINDOW	"window"
#define _TTYS_ONERROR   "onerror"
#define _TTYS_ONOPTION  "onoption"
#define _TTYS_SLOT      "slot"

struct ttyent {
	char	*ty_name;	/* terminal device name */
	char	*ty_getty;	/* command to execute, usually getty */
	char	*ty_type;	/* terminal type for termcap */
#define	TTY_ON		0x01	/* enable logins (start ty_getty program) */
#define	TTY_SECURE	0x02	/* allow uid of 0 to login */
	int	ty_status;	/* status flags */
	char 	*ty_window;	/* command to start up window manager */
	char    *ty_onerror;    /* command to execute after getty failure */
	char    *ty_onoption;   /* command to execute after console login */
	char	*ty_comment;	/* comment field */
};

#include <sys/cdefs.h>

__BEGIN_DECLS
struct ttyent *getttyent(void);
struct ttyent *getttynam(const char *);
int setttyent(void);
int endttyent(void);
__END_DECLS

#endif /* !_TTYENT_H_ */
