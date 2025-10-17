/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 12, 2024.
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
#ifndef _OSX_NTFS_UNISTR_H
#define _OSX_NTFS_UNISTR_H

#include "ntfs_layout.h"
#include "ntfs_types.h"
#include "ntfs_volume.h"

__private_extern__ BOOL ntfs_are_names_equal(const ntfschar *s1, size_t s1_len,
		const ntfschar *s2, size_t s2_len, const BOOL case_sensitive,
		const ntfschar *upcase, const u32 upcase_len);

__private_extern__ int ntfs_collate_names(const ntfschar *name1,
		const u32 name1_len, const ntfschar *name2,
		const u32 name2_len, const int err_val,
		const BOOL case_sensitive, const ntfschar *upcase,
		const u32 upcase_len);

__private_extern__ int ntfs_ucsncmp(const ntfschar *s1, const ntfschar *s2,
		size_t n);
__private_extern__ int ntfs_ucsncasecmp(const ntfschar *s1, const ntfschar *s2,
		size_t n, const ntfschar *upcase, const u32 upcase_size);

__private_extern__ void ntfs_upcase_name(ntfschar *name, u32 name_len,
		const ntfschar *upcase, const u32 upcase_len);

static inline void ntfs_file_upcase_value(FILENAME_ATTR *filename_attr,
		const ntfschar *upcase, const u32 upcase_len)
{
	ntfs_upcase_name(filename_attr->filename,
			filename_attr->filename_length, upcase, upcase_len);
}

static inline int ntfs_file_compare_values(FILENAME_ATTR *filename_attr1,
		FILENAME_ATTR *filename_attr2, const int err_val,
		const BOOL case_sensitive, const ntfschar *upcase,
		const u32 upcase_len)
{
	return ntfs_collate_names(filename_attr1->filename,
			filename_attr1->filename_length,
			filename_attr2->filename,
			filename_attr2->filename_length,
			err_val, case_sensitive, upcase, upcase_len);
}

__private_extern__ signed ntfs_to_utf8(const ntfs_volume *vol,
		const ntfschar *ins, const size_t ins_size,
		u8 **outs, size_t *outs_size);

__private_extern__ signed utf8_to_ntfs(const ntfs_volume *vol, const u8 *ins,
		const size_t ins_size, ntfschar **outs, size_t *outs_size);

__private_extern__ void ntfs_upcase_table_generate(ntfschar *uc, int uc_size);

#endif /* !_OSX_NTFS_UNISTR_H */
