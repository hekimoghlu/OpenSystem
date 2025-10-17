/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 12, 2024.
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
#ifndef ARCHIVE_XXHASH_H_INCLUDED
#define ARCHIVE_XXHASH_H_INCLUDED

#ifndef __LIBARCHIVE_BUILD
#error This header is only to be used internally to libarchive.
#endif


typedef enum { XXH_OK=0, XXH_ERROR } XXH_errorcode;

struct archive_xxhash {
	unsigned int  (*XXH32)(const void* input, unsigned int len,
			unsigned int seed);
	void*         (*XXH32_init)(unsigned int seed);
	XXH_errorcode (*XXH32_update)(void* state, const void* input,
			unsigned int len);
	unsigned int  (*XXH32_digest)(void* state);
};

extern const struct archive_xxhash __archive_xxhash;

#endif
