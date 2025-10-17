/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 10, 2022.
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
#ifndef _KXLD_SRCVERSION_H_
#define _KXLD_SRCVERSION_H_

#include <sys/types.h>
#if KERNEL
#include <libkern/kxld_types.h>
#else
#include "kxld_types.h"
#endif

struct source_version_command;
typedef struct kxld_srcversion KXLDsrcversion;

struct kxld_srcversion {
	uint64_t    version;
	boolean_t   has_srcversion;
};

/*******************************************************************************
* Constructors and destructors
*******************************************************************************/

void kxld_srcversion_init_from_macho(KXLDsrcversion *srcversion, struct source_version_command *src)
__attribute__((nonnull, visibility("hidden")));

void kxld_srcversion_clear(KXLDsrcversion *srcversion)
__attribute__((nonnull, visibility("hidden")));

/*******************************************************************************
* Accessors
*******************************************************************************/

u_long kxld_srcversion_get_macho_header_size(void)
__attribute__((pure, visibility("hidden")));

kern_return_t
kxld_srcversion_export_macho(const KXLDsrcversion *srcversion, u_char *buf,
    u_long *header_offset, u_long header_size)
__attribute__((pure, nonnull, visibility("hidden")));

#endif /* _KXLD_SRCVERSION_H_ */
