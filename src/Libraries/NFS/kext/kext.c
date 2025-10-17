/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 8, 2022.
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
 * Copyright (c) 1989, 1993, 1995
 *	The Regents of the University of California.  All rights reserved.
 *
 * This code is derived from software contributed to Berkeley by
 * Rick Macklem at The University of Guelph.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *	This product includes software developed by the University of
 *	California, Berkeley and its contributors.
 * 4. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#include "nfs_client.h"

#include <IOKit/IOLib.h>
#include <mach/mach_types.h>

kern_return_t
nfs_kext_start(kmod_info_t * ki, void *d)
{
	/*
	 * Register the memory zones.
	 */
	nfs_zone_init();

	/*
	 * Allocate global locks
	 */
	if (nfs_locks_init()) {
		goto fail;
	}

	/*
	 * Register sysctl on success
	 */
	nfs_sysctl_register();

	/*
	 * Register kernel hooks
	 */
	struct nfs_hooks_in hooks_in = { .f_vinvalbuf = nfs_vinvalbuf1, .f_buf_page_inval = nfs_buf_page_inval_internal };
	nfs_register_hooks(&hooks_in, &hooks_out);
	if (hooks_out.f_get_bsdthreadtask_info == NULL) {
		goto fail;
	}

	/*
	 * Add nfsclnt control device.
	 */
	if (nfsclnt_device_add()) {
		goto fail;
	}

	/*
	 * Register the file system.
	 */
	if (install_nfs_vfs_fs()) {
		goto fail;
	}

	IOLog("nfs_kext_start: successfully loaded NFS kext\n");
	return KERN_SUCCESS;

fail:
	nfsclnt_device_remove();
	nfs_locks_free();
	nfs_sysctl_unregister();
	nfs_zone_destroy();
	nfs_unregister_hooks();

	IOLog("nfs_kext_start: unable to load NFS kext\n");
	return KERN_FAILURE;
}

kern_return_t
nfs_kext_stop(kmod_info_t *ki, void *d)
{
	if (nfs_isbusy()) {
		return KERN_NO_ACCESS;
	}

	/*
	 * Unregister the file system.
	 */
	uninstall_nfs_vfs_fs();

	/*
	 * Remove nfsclnt control device.
	 */
	nfsclnt_device_remove();

	/*
	 * Unregister tsysctl.
	 */
	nfs_sysctl_unregister();

	/*
	 * Unregister the memory zones.
	 */
	nfs_zone_destroy();

	/*
	 * Release hash tables
	 */
	nfs_hashes_free();

	/*
	 * Termiante all active threads
	 */
	nfs_threads_terminate();

	/*
	 * Destroy global locks
	 */
	nfs_locks_free();

	/*
	 * Unregister kernel hooks
	 */
	nfs_unregister_hooks();

	IOLog("nfs_kext_stop: successfully stopped NFS kext\n");
	return KERN_SUCCESS;
}
