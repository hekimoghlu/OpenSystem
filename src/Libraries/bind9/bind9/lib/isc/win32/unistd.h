/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 24, 2025.
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
/* $Id: unistd.h,v 1.10 2009/07/17 23:47:41 tbox Exp $ */

/* None of these are defined in NT, so define them for our use */
#define O_NONBLOCK 1
#define PORT_NONBLOCK O_NONBLOCK

/*
 * fcntl() commands
 */
#define F_SETFL 0
#define F_GETFL 1
#define F_SETFD 2
#define F_GETFD 3
/*
 * Enough problems not having full fcntl() without worrying about this!
 */
#undef F_DUPFD

int fcntl(int, int, ...);

/*
 * access() related definitions for winXP
 */
#include <io.h>
#ifndef F_OK
#define	F_OK	0
#endif

#ifndef X_OK
#define	X_OK	1
#endif

#ifndef W_OK
#define W_OK 2
#endif

#ifndef R_OK
#define R_OK 4
#endif

#define access _access

#include <process.h>
