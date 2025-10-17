/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 2, 2024.
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
#ifndef _SYS_PGO_H_
#define _SYS_PGO_H_

#include <sys/_types.h>
#include <sys/_types/_ssize_t.h>
#include <stdint.h>
#include <uuid/uuid.h>

/* No longer supported. */
#define PGO_HIB (1)

#define PGO_WAIT_FOR_UNLOAD (2)
#define PGO_METADATA (4)
#define PGO_RESET_ALL (8)

#define PGO_ALL_FLAGS (PGO_HIB | PGO_WAIT_FOR_UNLOAD | PGO_METADATA | PGO_RESET_ALL)

/**
 * This is a serialization format for metadata related to a profile data buffer.
 *
 * If metadata is present, this footer will appear at the end of the file, so
 * the last four bytes of the file will be the ASCII string "meta".
 *
 * The metadata is stored in a environment-string style buffer.  The buffer
 * consists of key-value pairs, which are delimited by null bytes.  Each
 * key-value pair is a string of the form "FOO=bar".  Everything before the
 * first equal sign is the key, everything after is the value.
 *
 * All members are in network byte order.
 */
struct pgo_metadata_footer {
	/**
	 * number of pairs.
	 *
	 * This should be htonl(n), where n is the number of key-value pairs in the
	 * metadata buffer
	 */
	uint32_t number_of_pairs;

	/**
	 * pointer to the metadata buffer
	 *
	 * This should be htonl(offset), where offset is the backwards offset from
	 * the end of the file to the metadata buffer.
	 */
	uint32_t  offset_to_pairs;

	/**
	 * magic number
	 *
	 * This should be htonl(0x6d657461);
	 */
	uint32_t magic;
};

#ifndef KERNEL

ssize_t grab_pgo_data(
	uuid_t *uuid,
	int flags,
	unsigned char *buffer,
	ssize_t size);

#endif /* !defined(KERNEL) */

#ifdef XNU_KERNEL_PRIVATE
kern_return_t do_pgo_reset_counters(void);
#endif /* defined(XNU_KERNEL_PRIVATE) */

#endif /* !defined(XNU_KERNEL_PRIVATE) */
