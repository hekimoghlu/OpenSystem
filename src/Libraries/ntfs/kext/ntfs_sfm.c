/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 15, 2022.
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
#include "ntfs_endian.h"
#include "ntfs_sfm.h"
#include "ntfs_types.h"
#include "ntfs_unistr.h"
#include "ntfs_volume.h"

/*
 * A zerofilled finder info structure for fast checking of the finder info
 * being zero.
 */
const FINDER_INFO ntfs_empty_finder_info;

/*
 * These are the names used for the various named streams used by Services For
 * Macintosh (SFM) and the OS X SMB implementation and thus also by the OS X
 * NTFS driver.
 *
 * The following names are defined:
 *
 * 	AFP_AfpInfo
 * 	AFP_DeskTop
 * 	AFP_IdIndex
 * 	AFP_Resource
 * 	Comments
 *
 * See ntfs_sfm.h for further details.
 */

ntfschar NTFS_SFM_AFPINFO_NAME[12] = { const_cpu_to_le16('A'),
		const_cpu_to_le16('F'), const_cpu_to_le16('P'),
		const_cpu_to_le16('_'), const_cpu_to_le16('A'),
		const_cpu_to_le16('f'), const_cpu_to_le16('p'),
		const_cpu_to_le16('I'), const_cpu_to_le16('n'),
		const_cpu_to_le16('f'), const_cpu_to_le16('o'), 0 };

ntfschar NTFS_SFM_DESKTOP_NAME[12] = { const_cpu_to_le16('A'),
		const_cpu_to_le16('F'), const_cpu_to_le16('P'),
		const_cpu_to_le16('_'), const_cpu_to_le16('D'),
		const_cpu_to_le16('e'), const_cpu_to_le16('s'),
		const_cpu_to_le16('k'), const_cpu_to_le16('T'),
		const_cpu_to_le16('o'), const_cpu_to_le16('p'), 0 };

ntfschar NTFS_SFM_IDINDEX_NAME[12] = { const_cpu_to_le16('A'),
		const_cpu_to_le16('F'), const_cpu_to_le16('P'),
		const_cpu_to_le16('_'), const_cpu_to_le16('I'),
		const_cpu_to_le16('d'), const_cpu_to_le16('I'),
		const_cpu_to_le16('n'), const_cpu_to_le16('d'),
		const_cpu_to_le16('e'), const_cpu_to_le16('x'), 0 };

ntfschar NTFS_SFM_RESOURCEFORK_NAME[13] = { const_cpu_to_le16('A'),
		const_cpu_to_le16('F'), const_cpu_to_le16('P'),
		const_cpu_to_le16('_'), const_cpu_to_le16('R'),
		const_cpu_to_le16('e'), const_cpu_to_le16('s'),
		const_cpu_to_le16('o'), const_cpu_to_le16('u'),
		const_cpu_to_le16('r'), const_cpu_to_le16('c'),
		const_cpu_to_le16('e'), 0 };

ntfschar NTFS_SFM_COMMENTS_NAME[9] = { const_cpu_to_le16('C'),
		const_cpu_to_le16('o'), const_cpu_to_le16('m'),
		const_cpu_to_le16('m'), const_cpu_to_le16('e'),
		const_cpu_to_le16('n'), const_cpu_to_le16('t'),
		const_cpu_to_le16('s'), 0 };

/**
 * ntfs_is_sfm_name - check if a name is a protected SFM name
 * @name:	name (in NTFS Unicode) to check
 * @len:	length of name in NTFS Unicode characters to check
 *
 * Return true if the NTFS Unicode name @name of length @len characters is a
 * Services For Macintosh (SFM) protected name and false otherwise.
 */
BOOL ntfs_is_sfm_name(ntfs_volume *vol,
		const ntfschar *name, const unsigned len)
{
	const ntfschar *upcase = vol->upcase;
	const unsigned upcase_len = vol->upcase_len;
	const BOOL case_sensitive = NVolCaseSensitive(vol);

	return (ntfs_are_names_equal(name, len, NTFS_SFM_AFPINFO_NAME, 11,
			case_sensitive, upcase, upcase_len) ||
			ntfs_are_names_equal(name, len, NTFS_SFM_DESKTOP_NAME,
			11, case_sensitive, upcase, upcase_len) ||
			ntfs_are_names_equal(name, len, NTFS_SFM_IDINDEX_NAME,
			11, case_sensitive, upcase, upcase_len) ||
			ntfs_are_names_equal(name, len,
			NTFS_SFM_RESOURCEFORK_NAME, 12, case_sensitive, upcase,
			upcase_len) ||
			ntfs_are_names_equal(name, len, NTFS_SFM_COMMENTS_NAME,
			8, case_sensitive, upcase, upcase_len));
}
