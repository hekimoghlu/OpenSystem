/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 24, 2024.
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
#include <stdarg.h>
#include <sys/sem.h>

#include <errno.h>
/*
 * Because KERNEL is defined, including errno.h doesn't define errno, so
 * we have to do it ourselves.
 */
extern int * __error(void);
#define errno (*__error())

/*
 * Legacy stub to account for the differences in the ipc_perm structure,
 * while maintaining binary backward compatibility.
 */
extern int __semctl(int semid, int semnum, int cmd, void *);

int
semctl(int semid, int semnum, int cmd, ...)
{
	va_list			ap;
	int			rv;
	int			val = 0;
	struct __semid_ds_new	ds;
	struct __semid_ds_new	*ds_new = &ds;
	struct __semid_ds_old   *ds_old = NULL;

	va_start(ap, cmd);
	if (cmd == SETVAL)
		val = va_arg(ap, int);
	else
		ds_old = va_arg(ap, struct __semid_ds_old *);
	va_end(ap);

#define	_UP_CVT(x)	ds_new-> x = ds_old-> x
#define	_DN_CVT(x)	ds_old-> x = ds_new-> x

	if ((cmd == IPC_SET || cmd == IPC_STAT) && ds_old == NULL) {
		/* Return EFAULT if ds_old is NULL (like the kernel would do) */
		errno = EFAULT;
		return -1;
	}
	if (cmd == IPC_SET) {
		/* convert before call */
		_UP_CVT(sem_perm.uid);
		_UP_CVT(sem_perm.gid);
		_UP_CVT(sem_perm.cuid);
		_UP_CVT(sem_perm.cgid);
		_UP_CVT(sem_perm.mode);
		ds_new->sem_perm._seq = ds_old->sem_perm.seq;
		ds_new->sem_perm._key = ds_old->sem_perm.key;
		_UP_CVT(sem_base);
		_UP_CVT(sem_nsems);
		_UP_CVT(sem_otime);
		_UP_CVT(sem_pad1);	/* binary compatibility */
		_UP_CVT(sem_ctime);
		_UP_CVT(sem_pad2);	/* binary compatibility */
		_UP_CVT(sem_pad3[0]);	/* binary compatibility */
		_UP_CVT(sem_pad3[1]);	/* binary compatibility */
		_UP_CVT(sem_pad3[2]);	/* binary compatibility */
		_UP_CVT(sem_pad3[3]);	/* binary compatibility */
	}
	switch (cmd) {
	case SETVAL:
		/* syscall must use LP64 quantities */
		rv = __semctl(semid, semnum, cmd, (void *)val);
		break;
	case IPC_SET:
	case IPC_STAT:
		rv = __semctl(semid, semnum, cmd, ds_new);
		break;
	default:
		rv = __semctl(semid, semnum, cmd, ds_old);
		break;
	}

	if (cmd == IPC_STAT) {
		/* convert after call */
		_DN_CVT(sem_perm.uid);	/* warning!  precision loss! */
		_DN_CVT(sem_perm.gid);	/* warning!  precision loss! */
		_DN_CVT(sem_perm.cuid);	/* warning!  precision loss! */
		_DN_CVT(sem_perm.cgid);	/* warning!  precision loss! */
		_DN_CVT(sem_perm.mode);
		ds_old->sem_perm.seq = ds_new->sem_perm._seq;
		ds_old->sem_perm.key = ds_new->sem_perm._key;
		_DN_CVT(sem_base);
		_DN_CVT(sem_nsems);
		_DN_CVT(sem_otime);
		_DN_CVT(sem_pad1);	/* binary compatibility */
		_DN_CVT(sem_ctime);
		_DN_CVT(sem_pad2);	/* binary compatibility */
		_DN_CVT(sem_pad3[0]);	/* binary compatibility */
		_DN_CVT(sem_pad3[1]);	/* binary compatibility */
		_DN_CVT(sem_pad3[2]);	/* binary compatibility */
		_DN_CVT(sem_pad3[3]);	/* binary compatibility */
	}

	return (rv);
}
