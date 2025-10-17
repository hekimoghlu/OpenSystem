/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 8, 2021.
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
 *
 *	File: vm/vm_dyld_pager.h
 *
 *      protos and definitions for dyld pager
 */

#ifndef _VM_DYLD_PAGER_H_
#define _VM_DYLD_PAGER_H_

#include <mach/dyld_pager.h>

#ifdef KERNEL_PRIVATE
#include <kern/kern_types.h>
#include <vm/vm_map.h>

#define MWL_MIN_LINK_INFO_SIZE sizeof(struct mwl_info_hdr)
#define MWL_MAX_LINK_INFO_SIZE (64 * 1024 * 1024)   /* just a guess for now, may have to increase */

#endif /* KERNEL_PRIVATE */

#endif /* _VM_DYLD_PAGER_H_ */
