/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 26, 2021.
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
#ifndef _OSX_NTFS_VNOPS_H
#define _OSX_NTFS_VNOPS_H

#include <sys/buf.h>
#include <sys/ucred.h>
#include <sys/vnode.h>

typedef int vnop_t(void *);

__attribute__((visibility("hidden"))) extern vnop_t **ntfs_vnodeop_p;

__attribute__((visibility("hidden"))) extern struct vnodeopv_desc ntfs_vnodeopv_desc;

__private_extern__ int ntfs_cluster_iodone(buf_t cbp, void *arg __unused);

#endif /* !_OSX_NTFS_VNOPS_H */
