/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 26, 2024.
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
#ifndef _CACHE_H_
#define _CACHE_H_

/*
 * Constants and data structures used to implement group and password file
 * caches. Traditional passwd/group cache routines perform quite poorly with
 * archives. The chances of hitting a valid lookup with an archive is quite a
 * bit worse than with files already resident on the file system. These misses
 * create a MAJOR performance cost. To address this problem, these routines
 * cache both hits and misses.
 *
 * NOTE:  name lengths must be as large as those stored in ANY PROTOCOL and
 * as stored in the passwd and group files. CACHE SIZES MUST BE PRIME
 */
#define UNMLEN		32	/* >= user name found in any protocol */
#define GNMLEN		32	/* >= group name found in any protocol */
#define UID_SZ		317	/* size of user_name/uid cache */
#define UNM_SZ		317	/* size of user_name/uid cache */
#define GID_SZ		251	/* size of gid cache */
#define GNM_SZ		317	/* size of group name cache */
#define VALID		1	/* entry and name are valid */
#define INVALID		2	/* entry valid, name NOT valid */

/*
 * Node structures used in the user, group, uid, and gid caches.
 */

typedef struct uidc {
	int valid;		/* is this a valid or a miss entry */
	char name[UNMLEN];	/* uid name */
	uid_t uid;		/* cached uid */
} UIDC;

typedef struct gidc {
	int valid;		/* is this a valid or a miss entry */
	char name[GNMLEN];	/* gid name */
	gid_t gid;		/* cached gid */
} GIDC;

#endif /* _CACHE_H_ */
