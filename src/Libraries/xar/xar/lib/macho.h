/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 5, 2024.
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
 * Portions Copyright 2006, Apple Computer, Inc.
 * Christopher Ryan <ryanc@apple.com>
*/

#ifndef _MACHO_H_
#define _MACHO_H_

#include "xar.h"
#include "filetree.h"

struct mach_header {
   uint32_t magic;
   uint32_t cputype;
   uint32_t cpusubtype;
   uint32_t filetype;
   uint32_t ncmds;
   uint32_t sizeofcmds;
   uint32_t flags;
};

struct lc {
	uint32_t cmd;
	uint32_t cmdsize;
};

struct fat_header {
	uint32_t magic;
	uint32_t nfat_arch;
};

struct fat_arch {
	uint32_t cputype;
	uint32_t cpusubtype;
	uint32_t offset;
	uint32_t size;
	uint32_t alighn;
};

int32_t xar_macho_in(xar_t x, xar_file_t f, xar_prop_t p, void **in, size_t *inlen, void **context);
int32_t xar_macho_done(xar_t x, xar_file_t f, xar_prop_t p, void **context);

#endif /* _MACHO_H_ */
