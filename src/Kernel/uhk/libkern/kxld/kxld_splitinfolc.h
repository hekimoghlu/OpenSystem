/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 6, 2024.
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
#ifndef _KXLD_SPLITINFOLC_H_
#define _KXLD_SPLITINFOLC_H_

#include <sys/types.h>
#if KERNEL
#include <libkern/kxld_types.h>
#else
#include "kxld_types.h"
#endif

struct linkedit_data_command;
typedef struct kxld_splitinfolc KXLDsplitinfolc;

struct kxld_splitinfolc {
	uint32_t    cmdsize;
	uint32_t    dataoff;
	uint32_t    datasize;
	boolean_t   has_splitinfolc;
};

/*******************************************************************************
* Constructors and destructors
*******************************************************************************/

void kxld_splitinfolc_init_from_macho(KXLDsplitinfolc *splitinfolc, struct linkedit_data_command *src)
__attribute__((nonnull, visibility("hidden")));

void kxld_splitinfolc_clear(KXLDsplitinfolc *splitinfolc)
__attribute__((nonnull, visibility("hidden")));

/*******************************************************************************
* Accessors
*******************************************************************************/

u_long kxld_splitinfolc_get_macho_header_size(void)
__attribute__((pure, visibility("hidden")));

kern_return_t
kxld_splitinfolc_export_macho(const KXLDsplitinfolc *splitinfolc,
    splitKextLinkInfo *linked_object,
    u_long *header_offset,
    u_long header_size,
    u_long *data_offset,
    u_long size)
__attribute__((pure, nonnull, visibility("hidden")));

#endif /* _KXLD_SPLITINFOLC_H_ */
