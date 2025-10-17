/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 11, 2022.
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
 * Copyright (c) 1997 by Apple Computer, Inc., all rights reserved
 * Copyright (c) 1993 NeXT Computer, Inc.
 *
 */

#include <sys/param.h>
#include <sys/systm.h>
#include <sys/ioctl.h>
#include <sys/tty.h>
#include <sys/conf.h>
#include <sys/kauth.h>
#include <sys/ucred.h>
#include <sys/proc_internal.h>
#include <sys/user.h>
#include <kern/task.h>
#include <kern/thread.h>
#include <vm/vm_map.h>
#include <machine/machine_routines.h>
#include <sys/sysproto.h>

/*
 * copy a null terminated string from the kernel address space into
 * the user address space.
 *   - if the user is denied write access, return EFAULT.
 *   - if the end of string isn't found before
 *     maxlen bytes are copied,  return ENAMETOOLONG,
 *     indicating an incomplete copy.
 *   - otherwise, return 0, indicating success.
 * the number of bytes copied is always returned in lencopied.
 */
int
copyoutstr(const void *from, user_addr_t to, size_t maxlen, size_t *lencopied)
{
	size_t  slen;
	size_t  len;
	int     error = 0;

	slen = strlen(from) + 1;
	if (slen > maxlen) {
		error = ENAMETOOLONG;
	}

	len = min(maxlen, slen);
	if (copyout(from, to, len)) {
		error = EFAULT;
	}
	*lencopied = len;

	return error;
}


/*
 * copy a null terminated string from one point to another in
 * the kernel address space.
 *   - no access checks are performed.
 *   - if the end of string isn't found before
 *     maxlen bytes are copied,  return ENAMETOOLONG,
 *     indicating an incomplete copy.
 *   - otherwise, return 0, indicating success.
 * the number of bytes copied is always returned in lencopied.
 */
/* from ppc/fault_copy.c -Titan1T4 VERSION  */
int
copystr(const void *vfrom, void *vto, size_t maxlen, size_t *lencopied)
{
	size_t          l;
	char const      *from = (char const *) vfrom;
	char            *to = (char *) vto;

	for (l = 0; l < maxlen; l++) {
		if ((*to++ = *from++) == '\0') {
			if (lencopied) {
				*lencopied = l + 1;
			}
			return 0;
		}
	}
	if (lencopied) {
		*lencopied = maxlen;
	}
	return ENAMETOOLONG;
}

int
copywithin(void *src, void *dst, size_t count)
{
	bcopy(src, dst, count);
	return 0;
}

int
objc_bp_assist_cfg_np(
	__unused struct proc                        *p,
	__unused struct objc_bp_assist_cfg_np_args  *uap,
	__unused int                                *retvalp)
{
	return KERN_FAILURE;
}
