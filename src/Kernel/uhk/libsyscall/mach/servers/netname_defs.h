/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 22, 2025.
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
/*
 * Mach Operating System
 * Copyright (c) 1987 Carnegie-Mellon University
 * All rights reserved.  The CMU software License Agreement specifies
 * the terms and conditions for use and redistribution.
 */

/*
 * Definitions for the mig interface to the network name service.
 */

/*
 * HISTORY:
 * 28-Jul-88  Mary R. Thompson (mrt) at Carnegie Mellon
 *	Copied definitions of NAME_NOT_YOURS and NAME_NOT_CHECKED_IN
 *	from the old netname_defs.h so that old code would not break
 *
 *  8-Mar-88  Daniel Julin (dpj) at Carnegie-Mellon University
 *	Added NETNAME_INVALID_PORT.
 *
 * 28-Feb-88  Daniel Julin (dpj) at Carnegie-Mellon University
 *	Added NETNAME_PENDING.
 *
 * 23-Dec-86  Robert Sansom (rds) at Carnegie Mellon University
 *	Copied from the previous version of the network server.
 *
 */

#ifndef _NETNAME_DEFS_
#define _NETNAME_DEFS_

#define NETNAME_SUCCESS         (0)
#define NETNAME_PENDING         (-1)
#define NETNAME_NOT_YOURS       (1000)
#define NAME_NOT_YOURS          (1000)
#define NETNAME_NOT_CHECKED_IN  (1001)
#define NAME_NOT_CHECKED_IN     (1001)
#define NETNAME_NO_SUCH_HOST    (1002)
#define NETNAME_HOST_NOT_FOUND  (1003)
#define NETNAME_INVALID_PORT    (1004)

typedef char netname_name_t[80];

#endif /* NETNAME_DEFS_ */
