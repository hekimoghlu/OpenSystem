/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 18, 2025.
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
#ifndef _KXLD_UUID_H_
#define _KXLD_UUID_H_

#include <sys/types.h>
#if KERNEL
    #include <libkern/kxld_types.h>
#else
    #include "kxld_types.h"
#endif

struct uuid_command;
typedef struct kxld_uuid KXLDuuid;

struct kxld_uuid {
	u_char uuid[16];
	boolean_t has_uuid;
};

/*******************************************************************************
* Constructors and destructors
*******************************************************************************/

void kxld_uuid_init_from_macho(KXLDuuid *uuid, struct uuid_command *src)
__attribute__((nonnull, visibility("hidden")));

void kxld_uuid_clear(KXLDuuid *uuid)
__attribute__((nonnull, visibility("hidden")));

/*******************************************************************************
* Accessors
*******************************************************************************/

u_long kxld_uuid_get_macho_header_size(void)
__attribute__((pure, visibility("hidden")));

kern_return_t
kxld_uuid_export_macho(const KXLDuuid *uuid, u_char *buf,
    u_long *header_offset, u_long header_size)
__attribute__((pure, nonnull, visibility("hidden")));

#endif /* _KXLD_UUID_H_ */
