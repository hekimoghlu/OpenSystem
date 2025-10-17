/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 25, 2023.
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
 * NOTICE: This file was modified by McAfee Research in 2004 to introduce
 * support for mandatory and extensible security protections.  This notice
 * is included in support of clause 2.2 (b) of the Apple Public License,
 * Version 2.0.
 */
/*
 * Copyright (c) 1999 Apple Inc.  All rights reserved.
 *
 * HISTORY
 *
 * 1999 Mar 29 rsulack created.
 */

#include <mach/mach_types.h>
#include <mach/vm_types.h>
#include <mach/kern_return.h>
#include <mach/host_priv_server.h>
#include <mach/vm_map.h>

#include <kern/kern_types.h>
#include <kern/thread.h>

#include <vm/vm_kern.h>

#include <libkern/kernel_mach_header.h>

/*********************************************************************
 **********************************************************************
 ***           KMOD INTERFACE DEPRECATED AS OF SNOWLEOPARD          ***
 **********************************************************************
 **********************************************************************
 * Except for kmod_get_info(), which continues to work for K32 with
 * 32-bit clients, all remaining functions in this module remain
 * for symbol linkage or MIG support only,
 * and return KERN_NOT_SUPPORTED.
 *
 * Some kernel-internal portions have been moved to
 * libkern/OSKextLib.cpp and libkern/c++/OSKext.cpp.
 **********************************************************************/

// bsd/sys/proc.h
extern void proc_selfname(char * buf, int size);

#define NOT_SUPPORTED_USER64()    \
    do { \
	char procname[64] = "unknown";  \
	proc_selfname(procname, sizeof(procname));  \
	printf("%s is not supported for 64-bit clients (called from %s)\n",  \
	    __FUNCTION__, procname);  \
    } while (0)

#define NOT_SUPPORTED_KERNEL()    \
    do { \
	char procname[64] = "unknown";  \
	proc_selfname(procname, sizeof(procname));  \
	printf("%s is not supported on this kernel architecture (called from %s)\n",  \
	    __FUNCTION__, procname);  \
    } while (0)

#define KMOD_MIG_UNUSED __unused


/********************************************************************/
kern_return_t
kmod_get_info(
	host_t host __unused,
	kmod_info_array_t * kmod_list KMOD_MIG_UNUSED,
	mach_msg_type_number_t * kmodCount KMOD_MIG_UNUSED);
kern_return_t
kmod_get_info(
	host_t host __unused,
	kmod_info_array_t * kmod_list KMOD_MIG_UNUSED,
	mach_msg_type_number_t * kmodCount KMOD_MIG_UNUSED)
{
	NOT_SUPPORTED_KERNEL();
	return KERN_NOT_SUPPORTED;
}
