/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 1, 2022.
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
#include <unistd.h>
#include <sys/shm.h>

/*
 * Stub function to account for the differences in the ipc_perm structure,
 * while maintaining binary backward compatibility.
 *
 * This is only the legacy behavior.
 */
extern int __shmctl(int, int, void *);

int
shmctl(int shmid, int cmd, struct shmid_ds *ds)
{
	struct __shmid_ds_old	*ds_old = (struct __shmid_ds_old *)ds;
	struct __shmid_ds_new	ds2;
	struct __shmid_ds_new	*ds_new = &ds2;
	int			rv;

#define	_UP_CVT(x)	ds_new-> x = ds_old-> x
#define	_DN_CVT(x)	ds_old-> x = ds_new-> x

	if (cmd == IPC_SET) {
		/* convert before call */
		_UP_CVT(shm_perm.uid);
		_UP_CVT(shm_perm.gid);
		_UP_CVT(shm_perm.cuid);
		_UP_CVT(shm_perm.cgid);
		_UP_CVT(shm_perm.mode);
		ds_new->shm_perm._seq = ds_old->shm_perm.seq;
		ds_new->shm_perm._key = ds_old->shm_perm.key;
		_UP_CVT(shm_segsz);
		_UP_CVT(shm_lpid);
		_UP_CVT(shm_cpid);
		_UP_CVT(shm_nattch);
		_UP_CVT(shm_atime);
		_UP_CVT(shm_dtime);
		_UP_CVT(shm_ctime);
		_UP_CVT(shm_internal);
	}

	rv = __shmctl(shmid, cmd, (void *)ds_new);

	if (cmd == IPC_STAT) {
		/* convert after call */
		_DN_CVT(shm_perm.uid);	/* warning!  precision loss! */
		_DN_CVT(shm_perm.gid);	/* warning!  precision loss! */
		_DN_CVT(shm_perm.cuid);	/* warning!  precision loss! */
		_DN_CVT(shm_perm.cgid);	/* warning!  precision loss! */
		_DN_CVT(shm_perm.mode);
		ds_old->shm_perm.seq = ds_new->shm_perm._seq;
		ds_old->shm_perm.key = ds_new->shm_perm._key;
		_DN_CVT(shm_segsz);
		_DN_CVT(shm_lpid);
		_DN_CVT(shm_cpid);
		_DN_CVT(shm_nattch);
		_DN_CVT(shm_atime);
		_DN_CVT(shm_dtime);
		_DN_CVT(shm_ctime);
		_DN_CVT(shm_internal);
	}

	return (rv);
}
