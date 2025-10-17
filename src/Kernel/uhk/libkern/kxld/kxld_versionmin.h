/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 30, 2021.
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
#ifndef _KXLD_VERSIONMIN_H_
#define _KXLD_VERSIONMIN_H_

#include <sys/types.h>
#if KERNEL
    #include <libkern/kxld_types.h>
#else
    #include "kxld_types.h"
#endif

struct version_min_command;
typedef struct kxld_versionmin KXLDversionmin;

enum kxld_versionmin_platforms {
	kKxldVersionMinMacOSX,
	kKxldVersionMiniPhoneOS,
	kKxldVersionMinAppleTVOS,
	kKxldVersionMinWatchOS
};

struct kxld_versionmin {
	enum kxld_versionmin_platforms platform;
	uint32_t version;
	boolean_t has_versionmin;
};

/*******************************************************************************
* Constructors and destructors
*******************************************************************************/

void kxld_versionmin_init_from_macho(KXLDversionmin *versionmin, struct version_min_command *src)
__attribute__((nonnull, visibility("hidden")));

void kxld_versionmin_init_from_build_cmd(KXLDversionmin *versionmin, struct build_version_command *src)
__attribute__((nonnull, visibility("hidden")));

void kxld_versionmin_clear(KXLDversionmin *versionmin)
__attribute__((nonnull, visibility("hidden")));

/*******************************************************************************
* Accessors
*******************************************************************************/

u_long kxld_versionmin_get_macho_header_size(const KXLDversionmin *versionmin)
__attribute__((pure, visibility("hidden")));

kern_return_t
kxld_versionmin_export_macho(const KXLDversionmin *versionmin, u_char *buf,
    u_long *header_offset, u_long header_size)
__attribute__((pure, nonnull, visibility("hidden")));

#endif /* _KXLD_VERSIONMIN_H_ */
