/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 19, 2023.
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
//  edt_fstab.h
//
//  Created on 12/11/2018.
//

#ifndef edt_fstab_h
#define edt_fstab_h

#include <TargetConditionals.h>

#if (TARGET_OS_IPHONE && !TARGET_OS_SIMULATOR)

#include <stdlib.h>
#include <stdbool.h>

#define RAMDISK_FS_SPEC         "ramdisk"

/*
 *		get_boot_container, get_data_volume - return the bsd name of the requested
 *		device upon success. Null otherwise.
 */
const char          *get_boot_container(uint32_t *os_env);
const char          *get_data_volume(void);

int                 get_boot_manifest_hash(char *boot_manifest_hash, size_t boot_manifest_hash_len);

bool                enhanced_apfs_supported(void);
#endif

#endif /* edt_fstab_h */
