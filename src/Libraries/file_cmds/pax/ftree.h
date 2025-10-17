/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 19, 2023.
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
#ifndef _FTREE_H_
#define _FTREE_H_

/*
 * Data structure used by the ftree.c routines to store the file args to be
 * handed to fts(). It keeps a reference count of which args generated a
 * "selected" member
 */

typedef struct ftree {
	char		*fname;		/* file tree name */
	int		refcnt;		/* has tree had a selected file? */
	int		chflg;		/* change directory flag */
	struct ftree	*fow;		/* pointer to next entry on list */
} FTREE;

#endif /* _FTREE_H_ */
