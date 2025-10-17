/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 24, 2023.
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
#include <roken.h>

#include "asn1-common.h"
#include "check-common.h"

struct map_page {
    void *start;
    size_t size;
    void *data_start;
    size_t data_size;
    enum map_type type;
};

/* #undef HAVE_MMAP */

void *
map_alloc(enum map_type type, const void *buf,
	  size_t size, struct map_page **map)
{
#ifndef HAVE_MMAP
    unsigned char *p;
    size_t len = size + sizeof(long) * 2;
    int i;

    *map = calloc(1, sizeof(**map));
    if (*map == NULL)
	errx(1, "calloc");

    p = malloc(len);
    if (p == NULL)
	errx(1, "malloc");
    (*map)->type = type;
    (*map)->start = p;
    (*map)->size = len;
    (*map)->data_start = p + sizeof(long);
    for (i = sizeof(long); i > 0; i--)
	p[sizeof(long) - i] = 0xff - i;
    for (i = sizeof(long); i > 0; i--)
	p[len - i] = 0xff - i;
#else
    unsigned char *p;
    int flags, ret, fd;
    size_t pagesize = getpagesize();

    *map = calloc(1, sizeof(**map));
    if (*map == NULL)
	errx(1, "calloc");

    (*map)->type = type;

#ifdef MAP_ANON
    flags = MAP_ANON;
    fd = -1;
#else
    flags = 0;
    fd = open ("/dev/zero", O_RDONLY);
    if(fd < 0)
	err (1, "open /dev/zero");
#endif
    flags |= MAP_PRIVATE;

    (*map)->size = size + pagesize - (size % pagesize) + pagesize * 2;

    p = (unsigned char *)mmap(0, (*map)->size, PROT_READ | PROT_WRITE,
			      flags, fd, 0);
    if (p == (unsigned char *)MAP_FAILED)
	err (1, "mmap");

    (*map)->start = p;

    ret = mprotect (p, pagesize, 0);
    if (ret < 0)
	err (1, "mprotect");

    ret = mprotect (p + (*map)->size - pagesize, pagesize, 0);
    if (ret < 0)
	err (1, "mprotect");

    switch (type) {
    case OVERRUN:
	(*map)->data_start = p + (*map)->size - pagesize - size;
	break;
    case UNDERRUN:
	(*map)->data_start = p + pagesize;
	break;
    }
#endif
    (*map)->data_size = size;
    if (buf)
	memcpy((*map)->data_start, buf, size);
    return (*map)->data_start;
}

void
map_free(struct map_page *map, const char *test_name, const char *map_name)
{
#ifndef HAVE_MMAP
    unsigned char *p = map->start;
    int i;

    for (i = sizeof(long); i > 0; i--)
	if (p[sizeof(long) - i] != 0xff - i)
	    errx(1, "%s: %s underrun %d\n", test_name, map_name, i);
    for (i = sizeof(long); i > 0; i--)
	if (p[map->size - i] != 0xff - i)
	    errx(1, "%s: %s overrun %lu\n", test_name, map_name,
		 (unsigned long)map->size - i);
    free(map->start);
#else
    int ret;

    ret = munmap (map->start, map->size);
    if (ret < 0)
	err (1, "munmap");
#endif
    free(map);
}

static void
print_bytes (unsigned const char *buf, size_t len)
{
    size_t i;

    for (i = 0; i < len; ++i)
	printf ("%02x ", buf[i]);
}

#ifndef MAP_FAILED
#define MAP_FAILED (-1)
#endif

static char *current_test = "<uninit>";
static char *current_state = "<uninit>";

static RETSIGTYPE
segv_handler(int sig)
{
    int fd;
    char msg[] = "SIGSEGV i current test: ";

    fd = open("/dev/stdout", O_WRONLY, 0600);
    if (fd >= 0) {
	write(fd, msg, sizeof(msg));
	write(fd, current_test, strlen(current_test));
	write(fd, " ", 1);
	write(fd, current_state, strlen(current_state));
	write(fd, "\n", 1);
	close(fd);
    }
    _exit(1);
}

int
generic_test (const struct test_case *tests,
	      unsigned ntests,
	      size_t data_size,
	      int (ASN1CALL *encode)(unsigned char *, size_t, void *, size_t *),
	      int (ASN1CALL *length)(void *),
	      int (ASN1CALL *decode)(unsigned char *, size_t, void *, size_t *),
	      int (ASN1CALL *free_data)(void *),
	      int (*cmp)(void *a, void *b),
	      int (ASN1CALL *copy)(const void *from, void *to))
{
    unsigned char *buf, *buf2;
    unsigned i;
    int failures = 0;
    void *data;
    struct map_page *data_map, *buf_map, *buf2_map;

#ifdef HAVE_SIGACTION
    struct sigaction sa, osa;
#endif

    for (i = 0; i < ntests; ++i) {
	int ret;
	size_t sz, consumed_sz, length_sz, buf_sz;
	void *to = NULL;

	current_test = tests[i].name;

	current_state = "init";

#ifdef HAVE_SIGACTION
	sigemptyset (&sa.sa_mask);
	sa.sa_flags = 0;
#ifdef SA_RESETHAND
	sa.sa_flags |= SA_RESETHAND;
#endif
	sa.sa_handler = segv_handler;
	sigaction (SIGSEGV, &sa, &osa);
#endif

	data = map_alloc(OVERRUN, NULL, data_size, &data_map);

	buf_sz = tests[i].byte_len;
	buf = map_alloc(UNDERRUN, NULL, buf_sz, &buf_map);

	current_state = "encode";
	ret = (*encode) (buf + buf_sz - 1, buf_sz,
			 tests[i].val, &sz);
	if (ret != 0) {
	    printf ("encoding of %s failed %d\n", tests[i].name, ret);
	    ++failures;
	    continue;
	}
	if (sz != (size_t)tests[i].byte_len) {
 	    printf ("encoding of %s has wrong len (%lu != %lu)\n",
		    tests[i].name,
		    (unsigned long)sz, (unsigned long)tests[i].byte_len);
	    ++failures;
	    continue;
	}

	current_state = "length";
	length_sz = (*length) (tests[i].val);
	if (sz != length_sz) {
	    printf ("length for %s is bad (%lu != %lu)\n",
		    tests[i].name, (unsigned long)length_sz, (unsigned long)sz);
	    ++failures;
	    continue;
	}

	current_state = "memcmp";
	if (memcmp (buf, tests[i].bytes, tests[i].byte_len) != 0) {
	    printf ("encoding of %s has bad bytes:\n"
		    "correct: ", tests[i].name);
	    print_bytes ((unsigned char *)tests[i].bytes, tests[i].byte_len);
	    printf ("\nactual:  ");
	    print_bytes (buf, sz);
	    printf ("\n");
#if 0
	    rk_dumpdata("correct", tests[i].bytes, tests[i].byte_len);
	    rk_dumpdata("actual", buf, sz);
	    exit (1);
#endif
	    ++failures;
	    continue;
	}

	buf2 = map_alloc(OVERRUN, buf, sz, &buf2_map);

	current_state = "decode";
	ret = (*decode) (buf2, sz, data, &consumed_sz);
	if (ret != 0) {
	    printf ("decoding of %s failed %d\n", tests[i].name, ret);
	    ++failures;
	    continue;
	}
	if (sz != consumed_sz) {
	    printf ("different length decoding %s (%lu != %lu)\n",
		    tests[i].name,
		    (unsigned long)sz, (unsigned long)consumed_sz);
	    ++failures;
	    continue;
	}
	current_state = "cmp";
	if ((*cmp)(data, tests[i].val) != 0) {
	    printf ("%s: comparison failed\n", tests[i].name);
	    ++failures;
	    continue;
	}

	current_state = "copy";
	if (copy) {
	    to = malloc(data_size);
	    if (to == NULL)
		errx(1, "malloc");
	    ret = (*copy)(data, to);
	    if (ret != 0) {
		printf ("copy of %s failed %d\n", tests[i].name, ret);
		++failures;
		continue;
	    }

	    current_state = "cmp-copy";
	    if ((*cmp)(data, to) != 0) {
		printf ("%s: copy comparison failed\n", tests[i].name);
		++failures;
		continue;
	    }
	}

	current_state = "free";
	if (free_data) {
	    (*free_data)(data);
	    if (to) {
		(*free_data)(to);
		free(to);
	    }
	}

	current_state = "free";
	map_free(buf_map, tests[i].name, "encode");
	map_free(buf2_map, tests[i].name, "decode");
	map_free(data_map, tests[i].name, "data");

#ifdef HAVE_SIGACTION
	sigaction (SIGSEGV, &osa, NULL);
#endif
    }
    current_state = "done";
    return failures;
}

/*
 * check for failures
 *
 * a test size (byte_len) of -1 means that the test tries to trigger a
 * integer overflow (and later a malloc of to little memory), just
 * allocate some memory and hope that is enough for that test.
 */

int
generic_decode_fail (const struct test_case *tests,
		     unsigned ntests,
		     size_t data_size,
		     int (ASN1CALL *decode)(unsigned char *, size_t, void *, size_t *))
{
    unsigned char *buf;
    unsigned i;
    int failures = 0;
    void *data;
    struct map_page *data_map, *buf_map;

#ifdef HAVE_SIGACTION
    struct sigaction sa, osa;
#endif

    for (i = 0; i < ntests; ++i) {
	int ret;
	size_t sz;
	const void *bytes;

	current_test = tests[i].name;

	current_state = "init";

#ifdef HAVE_SIGACTION
	sigemptyset (&sa.sa_mask);
	sa.sa_flags = 0;
#ifdef SA_RESETHAND
	sa.sa_flags |= SA_RESETHAND;
#endif
	sa.sa_handler = segv_handler;
	sigaction (SIGSEGV, &sa, &osa);
#endif

	data = map_alloc(OVERRUN, NULL, data_size, &data_map);

	if (tests[i].byte_len < 0xffffff && tests[i].byte_len >= 0) {
	    sz = tests[i].byte_len;
	    bytes = tests[i].bytes;
	} else {
	    sz = 4096;
	    bytes = NULL;
	}

	buf = map_alloc(OVERRUN, bytes, sz, &buf_map);

	if (tests[i].byte_len == -1)
	    memset(buf, 0, sz);

	current_state = "decode";
	ret = (*decode) (buf, tests[i].byte_len, data, &sz);
	if (ret == 0) {
	    printf ("sucessfully decoded %s\n", tests[i].name);
	    ++failures;
	    continue;
	}

	current_state = "free";
	if (buf)
	    map_free(buf_map, tests[i].name, "encode");
	map_free(data_map, tests[i].name, "data");

#ifdef HAVE_SIGACTION
	sigaction (SIGSEGV, &osa, NULL);
#endif
    }
    current_state = "done";
    return failures;
}
