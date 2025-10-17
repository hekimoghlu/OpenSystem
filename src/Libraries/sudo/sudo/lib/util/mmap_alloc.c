/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 17, 2023.
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
 * This is an open source non-commercial project. Dear PVS-Studio, please check it.
 * PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
 */

#include <config.h>

#include <sys/types.h>
#include <sys/mman.h>

#include <errno.h>
#include <limits.h>
#include <string.h>
#if defined(HAVE_STDINT_H)
# include <stdint.h>
#elif defined(HAVE_INTTYPES_H)
# include <inttypes.h>
#endif

#include "sudo_compat.h"
#include "sudo_util.h"

#if !defined(MAP_ANON) && defined(MAP_ANONYMOUS)
# define MAP_ANON MAP_ANONYMOUS
#endif

#ifndef MAP_FAILED
# define MAP_FAILED ((void *)-1)
#endif

/*
 * This is sqrt(SIZE_MAX+1), as s1*s2 <= SIZE_MAX
 * if both s1 < MUL_NO_OVERFLOW and s2 < MUL_NO_OVERFLOW
 */
#define MUL_NO_OVERFLOW ((size_t)1 << (sizeof(size_t) * 4))

/*
 * Allocate "size" bytes via mmap().
 * Space is allocated to store the size for later unmapping.
 */
void *
sudo_mmap_alloc_v1(size_t size)
{
    void *ptr;
    unsigned long *ulp;
#ifndef MAP_ANON
    int fd;

    /* SunOS-style mmap allocation using /dev/zero. */
    if ((fd = open("/dev/zero", O_RDWR)) == -1)
	return NULL;
    size += sizeof(unsigned long);
    ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    close(fd);
#else
    size += sizeof(unsigned long);
    ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE|MAP_ANON, -1, 0);
#endif
    if (ptr == MAP_FAILED) {
	errno = ENOMEM;
	return NULL;
    }

    /* Store size before the actual data. */
    ulp = (unsigned long *)ptr;
    ulp[0] = size;
    return (void *)&ulp[1];
}

/*
 * Allocate "nmemb" elements of "size" bytes via mmap().
 * If overflow would occur, errno is set to ENOMEM and
 * NULL is returned.
 */
void *
sudo_mmap_allocarray_v1(size_t nmemb, size_t size)
{
    if ((nmemb >= MUL_NO_OVERFLOW || size >= MUL_NO_OVERFLOW) &&
	    nmemb > 0 && SIZE_MAX / nmemb < size) {
	errno = ENOMEM;
	return NULL;
    }
    return sudo_mmap_alloc_v1(nmemb * size);
}

/*
 * Make a copy of "str" via mmap() and return it.
 */
char *
sudo_mmap_strdup_v1(const char *str)
{
    size_t len = strlen(str);
    char *newstr;

    if (len == SIZE_MAX) {
	errno = ENOMEM;
	return NULL;
    }
    newstr = sudo_mmap_alloc_v1(len + 1);
    if (newstr != NULL) {
        memcpy(newstr, str, len);
        newstr[len] = '\0';
    }

    return newstr;
}

/*
 * Set the page permissions for the allocation represented by "ptr" to
 * read-only.  Returns 0 on success, -1 on failure.
 */
int
sudo_mmap_protect_v1(void *ptr)
{
    if (ptr != NULL) {
	unsigned long *ulp = ptr;
	const unsigned long size = ulp[-1];
	return mprotect((void *)&ulp[-1], size, PROT_READ);
    }

    /* Can't protect NULL. */
    errno = EINVAL;
    return -1;
}

/*
 * Free "ptr" allocated by sudo_mmap_alloc().
 * The allocated size is stored (as unsigned long) in ptr[-1].
 */
void
sudo_mmap_free_v1(void *ptr)
{
    if (ptr != NULL) {
	unsigned long *ulp = (unsigned long *)ptr - 1;
	const unsigned long size = ulp[0];
	int saved_errno = errno;

	(void)munmap((void *)ulp, size);
	errno = saved_errno;
    }
}
