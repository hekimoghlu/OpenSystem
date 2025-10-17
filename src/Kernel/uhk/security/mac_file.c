/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 17, 2022.
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
#include <sys/param.h>
#include <sys/kernel.h>
#include <sys/lock.h>
#include <sys/malloc.h>
#include <sys/proc.h>
#include <sys/sbuf.h>
#include <sys/systm.h>
#include <sys/vnode.h>
#include <sys/vnode_internal.h>
#include <sys/file.h>
#include <sys/file_internal.h>

#include <security/mac_internal.h>

int
mac_file_check_create(struct ucred *cred)
{
	int error;

	MAC_CHECK(file_check_create, cred);
	return error;
}

int
mac_file_check_dup(struct ucred *cred, struct fileglob *fg, int newfd)
{
	int error;

	MAC_CHECK(file_check_dup, cred, fg, NULL, newfd);
	return error;
}

int
mac_file_check_fcntl(struct ucred *cred, struct fileglob *fg, int cmd,
    user_long_t arg)
{
	int error;

	MAC_CHECK(file_check_fcntl, cred, fg, NULL, cmd, arg);
	return error;
}

int
mac_file_check_ioctl(struct ucred *cred, struct fileglob *fg, u_long cmd)
{
	int error;

	MAC_CHECK(file_check_ioctl, cred, fg, NULL, cmd);
	return error;
}

int
mac_file_check_inherit(struct ucred *cred, struct fileglob *fg)
{
	int error;

	MAC_CHECK(file_check_inherit, cred, fg, NULL);
	return error;
}

int
mac_file_check_receive(struct ucred *cred, struct fileglob *fg)
{
	int error;

	MAC_CHECK(file_check_receive, cred, fg, NULL);
	return error;
}

int
mac_file_check_get_offset(struct ucred *cred, struct fileglob *fg)
{
	int error;

	MAC_CHECK(file_check_get_offset, cred, fg, NULL);
	return error;
}

int
mac_file_check_change_offset(struct ucred *cred, struct fileglob *fg)
{
	int error;

	MAC_CHECK(file_check_change_offset, cred, fg, NULL);
	return error;
}

int
mac_file_check_get(struct ucred *cred, struct fileglob *fg, char *elements,
    size_t len)
{
	int error;

	MAC_CHECK(file_check_get, cred, fg, elements, len);
	return error;
}

int
mac_file_check_set(struct ucred *cred, struct fileglob *fg, char *buf,
    size_t buflen)
{
	int error;

	MAC_CHECK(file_check_set, cred, fg, buf, buflen);
	return error;
}

int
mac_file_check_lock(struct ucred *cred, struct fileglob *fg, int op,
    struct flock *fl)
{
	int error;

	MAC_CHECK(file_check_lock, cred, fg, NULL, op, fl);
	return error;
}

int
mac_file_check_library_validation(struct proc *proc,
    struct fileglob *fg, off_t slice_offset,
    user_long_t error_message, size_t error_message_size)
{
	int error;

	MAC_CHECK(file_check_library_validation, proc, fg, slice_offset, error_message, error_message_size);
	return error;
}

/*
 * On some platforms, VM_PROT_READ implies VM_PROT_EXECUTE. If that is true,
 * both prot and maxprot will have VM_PROT_EXECUTE set after file_check_mmap
 * if VM_PROT_READ is set.
 *
 * The type of maxprot in file_check_mmap must be equivalent to vm_prot_t *
 * (defined in <mach/vm_prot.h>). mac_policy.h does not include any header
 * files, so cannot use the typedef itself.
 */
int
mac_file_check_mmap(struct ucred *cred, struct fileglob *fg, int prot,
    int flags, uint64_t offset, int *maxprot)
{
	int error;
	int maxp;

	maxp = *maxprot;
	MAC_CHECK(file_check_mmap, cred, fg, NULL, prot, flags, offset, &maxp);
	if ((maxp | *maxprot) != *maxprot) {
		panic("file_check_mmap increased max protections");
	}
	*maxprot = maxp;
	return error;
}

void
mac_file_check_mmap_downgrade(struct ucred *cred, struct fileglob *fg,
    int *prot)
{
	int result = *prot;

	MAC_PERFORM(file_check_mmap_downgrade, cred, fg, NULL, &result);

	*prot = result;
}

void
mac_file_notify_close(struct ucred *cred, struct fileglob *fg)
{
	MAC_PERFORM(file_notify_close, cred, fg, NULL, ((fg->fg_flag & FWASWRITTEN) ? 1 : 0));
}


/*
 * fileglob XATTR helpers.
 */

int
mac_file_setxattr(struct fileglob *fg, const char *name, char *buf, size_t len)
{
	struct vnode *vp = NULL;

	if (!fg || FILEGLOB_DTYPE(fg) != DTYPE_VNODE) {
		return EFTYPE;
	}

	vp = (struct vnode *)fg_get_data(fg);
	int error = vnode_getwithref(vp);
	if (error) {
		return error;
	}
	error = mac_vnop_setxattr(vp, name, buf, len);
	vnode_put(vp);
	return error;
}

int
mac_file_getxattr(struct fileglob *fg, const char *name, char *buf, size_t len,
    size_t *attrlen)
{
	struct vnode *vp = NULL;

	if (!fg || FILEGLOB_DTYPE(fg) != DTYPE_VNODE) {
		return EFTYPE;
	}

	vp = (struct vnode *)fg_get_data(fg);
	int error = vnode_getwithref(vp);
	if (error) {
		return error;
	}
	error = mac_vnop_getxattr(vp, name, buf, len, attrlen);
	vnode_put(vp);
	return error;
}

int
mac_file_removexattr(struct fileglob *fg, const char *name)
{
	struct vnode *vp = NULL;

	if (!fg || FILEGLOB_DTYPE(fg) != DTYPE_VNODE) {
		return EFTYPE;
	}

	vp = (struct vnode *)fg_get_data(fg);
	int error = vnode_getwithref(vp);
	if (error) {
		return error;
	}
	error = mac_vnop_removexattr(vp, name);
	vnode_put(vp);
	return error;
}
