/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 3, 2025.
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
#ifndef _SEL_SUBS_H_
#define _SEL_SUBS_H_

/*
 * data structure for storing uid/grp selects (-U, -G non standard options)
 */

#define USR_TB_SZ	317		/* user selection table size */
#define GRP_TB_SZ	317		/* user selection table size */

typedef struct usrt {
	uid_t uid;
	struct usrt *fow;		/* next uid */
} USRT;

typedef struct grpt {
	gid_t gid;
	struct grpt *fow;		/* next gid */
} GRPT;

/*
 * data structure for storing user supplied time ranges (-T option)
 */

#define ATOI2(ar)	((ar)[0] - '0') * 10 + ((ar)[1] - '0'); (ar) += 2;

typedef struct time_rng {
	time_t		low_time;	/* lower inclusive time limit */
	time_t		high_time;	/* higher inclusive time limit */
	int		flgs;		/* option flags */
#define	HASLOW		0x01		/* has lower time limit */
#define HASHIGH		0x02		/* has higher time limit */
#define CMPMTME		0x04		/* compare file modification time */
#define CMPCTME		0x08		/* compare inode change time */
#define CMPBOTH	(CMPMTME|CMPCTME)	/* compare inode and mod time */
	struct time_rng	*fow;		/* next pattern */
} TIME_RNG;

#endif /* _SEL_SUBS_H_ */
