/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 29, 2024.
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
 * nbsp.h
 * - NetBoot SharePoint routines
 */

#ifndef _S_NBSP_H
#define _S_NBSP_H

#include <stdbool.h>

#define NBSP_NO_READONLY	FALSE
#define NBSP_READONLY_OK	TRUE

typedef struct {
    char *	name;
    char *	path;
    bool	is_hfs;
    bool	is_readonly;
} NBSPEntry, * NBSPEntryRef;

struct NBSPList_s;

typedef struct NBSPList_s * NBSPListRef;

int		NBSPList_count(NBSPListRef list);
NBSPEntryRef	NBSPList_element(NBSPListRef list, int i);
void		NBSPList_print(NBSPListRef list);
void		NBSPList_free(NBSPListRef * list);
NBSPListRef	NBSPList_init(const char * symlink_name, bool readonly_ok);

#endif /* _S_NBSP_H */
