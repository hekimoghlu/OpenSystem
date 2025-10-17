/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 7, 2024.
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
#pragma prototyped

#include "asohdr.h"

#if defined(_UWIN) && defined(_BLD_ast) || !_aso_semaphore

NoN(aso_meth_semaphore)

#else

#include <sys/stat.h>
#include <sys/ipc.h>
#include <sys/sem.h>

#define SPIN		1000000

typedef union Semun_u
{
	int			val;
	struct semid_ds*	ds;
	unsigned short*		array;
} Semun_t;

typedef struct APL_s
{
	int		id;
	size_t		size;
} APL_t;

static void*
aso_init_semaphore(void* data, const char* details)
{
	APL_t*		apl = (APL_t*)data;
	char*		path;
	char*		opt;
	size_t		size;
	size_t		n;
	int		key;
	int		id;
	int		perm;
	struct sembuf	sem;
	char		tmp[64];

	if (apl)
	{
		/*
		 * semaphore 0 is the reference count
		 * the id is dropped on last reference
		 */

		sem.sem_num = 0;
		sem.sem_op = -1;
		sem.sem_flg = IPC_NOWAIT;
		semop(apl->id, &sem, 1);
		sem.sem_op = 0;
		if (!semop(apl->id, &sem, 1))
			semctl(apl->id, 0, IPC_RMID);
		free(apl);
		return 0;
	}
	perm = S_IRUSR|S_IWUSR;
	size = 128;
	if (path = (char*)details)
		while (opt = strchr(path, ','))
		{
			if (strneq(path, "perm=", 5))
			{
				if ((n = opt - (path + 5)) >= sizeof(tmp))
					n = sizeof(tmp) - 1;
				memcpy(tmp, path + 5, n);
				tmp[n] = 0;
				perm = strperm(tmp, NiL, perm);
			}
			else if (strneq(path, "size=", 5))
			{
				size = strtoul(path + 5, NiL, 0);
				if (size <= 1)
					return 0;
			}
			path = opt + 1;
		}
	key = (!path || !*path || streq(path, "private")) ? IPC_PRIVATE : (strsum(path, 0) & 0x7fff);
	for (;;)
	{	
		if ((id = semget(key, size, IPC_CREAT|IPC_EXCL|perm)) >= 0)
		{
			/*
			 * initialize all semaphores to 0
			 * this also sets the semaphore 0 ref count
			 */

			sem.sem_op = 1;
			sem.sem_flg = 0;
			for (sem.sem_num = 0; sem.sem_num < size; sem.sem_num++)
				if (semop(id, &sem, 1) < 0)
				{	
					(void)semctl(id, 0, IPC_RMID);
					return 0;
				}
			break;
		}
		else if (errno == EINVAL && size > 3)
			size /= 2;
		else if (errno != EEXIST)
			return 0;
		else if ((id = semget(key, size, perm)) >= 0)
		{	
			struct semid_ds	ds;
			Semun_t		arg;
			unsigned int	k;

			/*
			 * make sure all semaphores have been activated
			 */

			arg.ds = &ds;
			for (k = 0; k < SPIN; ASOLOOP(k))
			{	
				if (semctl(id, size-1, IPC_STAT, arg) < 0)
					return 0;
				if (ds.sem_otime)
					break;
			}
			if (k > SPIN)
				return 0;

			/*
			 * bump the ref count
			 */

			sem.sem_num = 0;
			sem.sem_op = 1;
			sem.sem_flg = 0;
			if (semop(id, &sem, 1) < 0)
				return 0;
			break;
		}
		else if (errno == EINVAL && size > 3)
			size /= 2;
		else
			return 0;
	}
	if (!(apl = newof(0, APL_t, 1, 0)))
		return 0;
	apl->id = id;
	apl->size = size - 1;
	return apl;
}

static ssize_t
aso_lock_semaphore(void* data, ssize_t k, void volatile* p)
{
	APL_t*		apl = (APL_t*)data;
	struct sembuf	sem;

	if (!apl)
		return -1;
	if (k > 0)
		sem.sem_op = 1;
	else
	{
		sem.sem_op = -1;
		k = HASH(p, apl->size) + 1;
	}
	sem.sem_num = k;
	sem.sem_flg = 0;
	return semop(apl->id, &sem, 1) < 0 ? -1 : k;
}

Asometh_t	_aso_meth_semaphore = { "semaphore", ASO_PROCESS|ASO_THREAD, aso_init_semaphore, aso_lock_semaphore };

#endif
