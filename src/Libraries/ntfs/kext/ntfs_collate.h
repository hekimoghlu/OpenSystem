/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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
#ifndef _OSX_NTFS_COLLATE_H
#define _OSX_NTFS_COLLATE_H

#include "ntfs_layout.h"
#include "ntfs_types.h"
#include "ntfs_volume.h"

static inline BOOL ntfs_is_collation_rule_supported(COLLATION_RULE cr) {
	int i;

	/*
	 * TODO: We support everything other than COLLATION_UNICODE_STRING at
	 * present but we do a range check in case new collation rules turn up
	 * in later ntfs releases.
	 */
	if (cr == COLLATION_UNICODE_STRING)
		return FALSE;
	i = le32_to_cpu(cr);
	if (((i >= 0) && (i <= 0x02)) || ((i >= 0x10) && (i <= 0x13)))
		return TRUE;
	return FALSE;
}

__private_extern__ int ntfs_collate(ntfs_volume *vol, COLLATION_RULE cr,
		const void *data1, const int data1_len,
		const void *data2, const int data2_len);

#endif /* _OSX_NTFS_COLLATE_H */
