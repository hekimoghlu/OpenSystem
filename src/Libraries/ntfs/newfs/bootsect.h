/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 4, 2022.
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
#ifndef _NTFS_BOOTSECT_H
#define _NTFS_BOOTSECT_H

#include "types.h"
#include "layout.h"

/**
 * ntfs_boot_sector_is_ntfs - check a boot sector for describing an ntfs volume
 * @b:		buffer containing the boot sector
 *
 * This function checks the boot sector in @b for describing a valid ntfs
 * volume. Return TRUE if @b is a valid NTFS boot sector or FALSE otherwise.
 */
extern BOOL ntfs_boot_sector_is_ntfs(NTFS_BOOT_SECTOR *b);

#endif /* defined _NTFS_BOOTSECT_H */

