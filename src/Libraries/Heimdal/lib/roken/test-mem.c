/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 18, 2025.
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
#include <config.h>

#ifdef HAVE_SYS_MMAN_H
#include <sys/mman.h>
#endif
#include <stdio.h>
#include <string.h>
#include <err.h>
#include "roken.h"

#include "test-mem.h"

/* #undef HAVE_MMAP */

struct {
    void *start;
    size_t size;
    void *data_start;
    size_t data_size;
    enum rk_test_mem_type type;
    int fd;
} map;

#ifdef HAVE_SIGACTION

struct sigaction sa, osa;

#else

void (* osigh)(int);

#endif

char *testname;

static RETSIGTYPE
segv_handler(int sig)
{
    int fd;
    char msg[] = "SIGSEGV i current test: ";

    fd = open("/dev/stdout", O_WRONLY, 0600);
    if (fd >= 0) {
	(void)write(fd, msg, sizeof(msg) - 1);
	(void)write(fd, testname, strlen(testname));
	(void)write(fd, "\n", 1);
	close(fd);
    }
    _exit(1);
}

#define TESTREC()							\
    if (testname)							\
	errx(1, "test %s run recursively on %s", name, testname);	\
    testname = strdup(name);						\
    if (testname == NULL)						\
	errx(1, "malloc");


ROKEN_LIB_FUNCTION void * ROKEN_LIB_CALL
rk_test_mem_alloc(enum rk_test_mem_type type, const char *name,
		  void *buf, size_t size)
{
#ifndef HAVE_MMAP
    unsigned char *p;

    TESTREC();

    p = malloc(size + 2);
    if (p == NULL)
	errx(1, "malloc");
    map.type = type;
    map.start = p;
    map.size = size + 2;
    p[0] = 0xff;
    p[map.size-1] = 0xff;
    map.data_start = p + 1;
#else
    unsigned char *p;
    int flags, ret, fd;
    size_t pagesize = getpagesize();

    TESTREC();

    map.type = type;

#ifdef MAP_ANON
    flags = MAP_ANON;
    fd = -1;
#else
    flags = 0;
    fd = open ("/dev/zero", O_RDONLY);
    if(fd < 0)
	err (1, "open /dev/zero");
#endif
    map.fd = fd;
    flags |= MAP_PRIVATE;

    map.size = size + pagesize - (size % pagesize) + pagesize * 2;

    p = (unsigned char *)mmap(0, map.size, PROT_READ | PROT_WRITE,
			      flags, fd, 0);
    if (p == (unsigned char *)MAP_FAILED)
	err (1, "mmap");

    map.start = p;

    ret = mprotect ((void *)p, pagesize, 0);
    if (ret < 0)
	err (1, "mprotect");

    ret = mprotect (p + map.size - pagesize, pagesize, 0);
    if (ret < 0)
	err (1, "mprotect");

    switch (type) {
    case RK_TM_OVERRUN:
	map.data_start = p + map.size - pagesize - size;
	break;
    case RK_TM_UNDERRUN:
	map.data_start = p + pagesize;
	break;
    default:
	abort();
    }
#endif
#ifdef HAVE_SIGACTION
    sigemptyset (&sa.sa_mask);
    sa.sa_flags = 0;
#ifdef SA_RESETHAND
    sa.sa_flags |= SA_RESETHAND;
#endif
    sa.sa_handler = segv_handler;
    sigaction (SIGSEGV, &sa, &osa);
#else
    osigh = signal(SIGSEGV, segv_handler);
#endif

    map.data_size = size;
    if (buf)
	memcpy(map.data_start, buf, size);
    return map.data_start;
}

ROKEN_LIB_FUNCTION void ROKEN_LIB_CALL
rk_test_mem_free(const char *map_name)
{
#ifndef HAVE_MMAP
    unsigned char *p = map.start;

    if (testname == NULL)
	errx(1, "test_mem_free call on no free");

    if (p[0] != 0xff)
	errx(1, "%s: %s underrun %x\n", testname, map_name, p[0]);
    if (p[map.size-1] != 0xff)
	errx(1, "%s: %s overrun %x\n", testname, map_name, p[map.size - 1]);
    free(map.start);
#else
    int ret;

    if (testname == NULL)
	errx(1, "test_mem_free call on no free");

    ret = munmap (map.start, map.size);
    if (ret < 0)
	err (1, "munmap");
    if (map.fd > 0)
	close(map.fd);
#endif
    free(testname);
    testname = NULL;

#ifdef HAVE_SIGACTION
    sigaction (SIGSEGV, &osa, NULL);
#else
    signal (SIGSEGV, osigh);
#endif
}
