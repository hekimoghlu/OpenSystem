/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 5, 2025.
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
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "param.h"
#include "types.h"
#include "debug.h"
#include "attrib.h"
#include "inode.h"
#include "dir.h"

/*
 * The little endian Unicode strings "$I30", "$SII", "$SDH", "$O"
 *  and "$Q" as global constants.
 */
ntfschar NTFS_INDEX_I30[5] = { const_cpu_to_le16('$'), const_cpu_to_le16('I'),
		const_cpu_to_le16('3'), const_cpu_to_le16('0'),
		const_cpu_to_le16('\0') };
ntfschar NTFS_INDEX_SII[5] = { const_cpu_to_le16('$'), const_cpu_to_le16('S'),
		const_cpu_to_le16('I'), const_cpu_to_le16('I'),
		const_cpu_to_le16('\0') };
ntfschar NTFS_INDEX_SDH[5] = { const_cpu_to_le16('$'), const_cpu_to_le16('S'),
		const_cpu_to_le16('D'), const_cpu_to_le16('H'),
		const_cpu_to_le16('\0') };
ntfschar NTFS_INDEX_O[3] = { const_cpu_to_le16('$'), const_cpu_to_le16('O'),
		const_cpu_to_le16('\0') };
ntfschar NTFS_INDEX_Q[3] = { const_cpu_to_le16('$'), const_cpu_to_le16('Q'),
		const_cpu_to_le16('\0') };
ntfschar NTFS_INDEX_R[3] = { const_cpu_to_le16('$'), const_cpu_to_le16('R'),
		const_cpu_to_le16('\0') };
