/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 8, 2022.
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
#ifndef _VFS_DISK_CONDITIONER_H_
#define _VFS_DISK_CONDITIONER_H_

#ifdef KERNEL_PRIVATE

#include <sys/fsctl.h>
int disk_conditioner_get_info(mount_t, disk_conditioner_info *);
int disk_conditioner_set_info(mount_t, disk_conditioner_info *);

boolean_t disk_conditioner_mount_is_ssd(mount_t);

#endif /* KERNEL_PRIVATE */

#endif /* !_VFS_DISK_CONDITIONER_H_ */
