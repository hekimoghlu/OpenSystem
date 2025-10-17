/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 7, 2022.
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
#include <sys/ioctl.h>
#include <termios.h>

struct info {
	int fd;					/* file descriptor */
	int ldisc;				/* line discipline */
	int off;				/* turn off */
	int set;				/* need set */
	int wset;				/* need window set */
	const char *arg;			/* argument */
	struct termios t;			/* terminal info */
	struct winsize win;			/* window info */
};

struct cchar {
	const char *name;
	int sub;
	u_char def;
};

enum FMT { NOTSET, GFLAG, BSD, POSIX };

#define	LINELENGTH	72
