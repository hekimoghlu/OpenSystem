/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 6, 2022.
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
#ifndef _DOUBLEAGENT_TYPES_H_
#define _DOUBLEAGENT_TYPES_H_

#define DA_XATTR_MAXNAMELEN 127 // Must match the 'XATTR_MAXNAMELEN' in <sys/xattr.h>.
#define DA_XATTR_FINDERINFO_NAME "com.apple.FinderInfo" // Copy of XATTR_FINDERINFO_NAME in <sys/xattr.h>.
#define DA_XATTR_RESOURCEFORK_NAME "com.apple.ResourceFork" // Copy of DA_XATTR_RESOURCEFORK_NAME in <sys/xattr.h>.

#define MAX_NUM_OF_XATTRS 256
#define LISTXATTR_RESULT_MAX_NAMES_LEN (sizeof(DA_XATTR_RESOURCEFORK_NAME) + sizeof(DA_XATTR_FINDERINFO_NAME) + (MAX_NUM_OF_XATTRS * ((DA_XATTR_MAXNAMELEN + 1))))
#define LISTXATTR_RESULT_MAX_HINTS_LEN (MAX_NUM_OF_XATTRS * 2 * sizeof(uint32_t)) // hint = offset + length (per xattr).
#define LISTXATTR_RESULT_MAX_SIZE (LISTXATTR_RESULT_MAX_NAMES_LEN + LISTXATTR_RESULT_MAX_HINTS_LEN)

typedef char xattrname[DA_XATTR_MAXNAMELEN + 1];

typedef struct list_xattrs_result {
	/* header */
	uint64_t finderInfoOffset; // =0 if not present
	uint64_t resourceForkOffset; // =0 if not present
	uint64_t resourceForkLength; // Don't care if resourceForkOffset = 0
	uint64_t numOfXattrs;

	/* data:
	 * (1) names (separated with '\0')
	 * (2) ranges: offset + lengths (for caching)
	 * (dataLength = namesLength + rangesLength)
	 */
	uint64_t dataLength;
	uint64_t namesLength;
	uint64_t rangesLength;
	uint8_t  data[LISTXATTR_RESULT_MAX_SIZE];
} listxattrs_result_t;

#endif /* _DOUBLEAGENT_TYPES_H_ */
