/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 20, 2024.
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
#include <sys/errno.h>
#include <sys/stat.h>

#include <ctype.h>
#include <dirent.h>
#include <dlfcn.h>
#include <err.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "check-common.h"
#include "fuzzer.h"

static char *leak_cmd = NULL;
static void *dso;

unsigned long count = 0;

static void
runTestcase(const char *name, const unsigned char *p, size_t length, heim_fuzz_type_t type)
{
    int (*decode_item)(const unsigned char *, size_t, void *, size_t *);
    int (*free_item)(void *);
    size_t (*size_item)(void);
    char *decode_name, *free_name, *size_name;
    size_t size;
    void *data;
    int ret;
    unsigned char *copy;
    unsigned long tcount = 1;
    unsigned long lcount = 0;
    size_t datasize = 10000;
    struct map_page *data_map, *copy_map;
    void *ctx = NULL;

    if (type) {
	printf("fuzzing using: %s\n", heim_fuzzer_name(type));
	tcount = 10000000;
    } else {
	printf("non fuzzings\n");
    }

    asprintf(&decode_name, "decode_%s", name);
    asprintf(&free_name, "free_%s", name);
    asprintf(&size_name, "size_%s", name);
    
    decode_item = dlsym(dso, decode_name);
    free_item = dlsym(dso, free_name);
    size_item = dlsym(dso, size_name);

    free(decode_name);
    free(free_name);
    
    if (decode_item == NULL)
	errx(1, "no decode_%s", name);
    if (free_item == NULL)
	errx(1, "no free_%s", name);

    /* should export size_encoder */
    if (size_item)
	datasize = size_item();
    else
	datasize = 10000;


    data = map_alloc(OVERRUN, NULL, datasize, &data_map);
    memset(data, 0, datasize);

    copy = map_alloc(OVERRUN, NULL, length, &copy_map);
    memset(copy, 0, length);

    /*
     * Main fuzzer loop, keep modifying the input stream as long as it
     * parses clearly.
     */

    memcpy(copy, p, length);
    while (tcount > 0) {

	if (type) {
	    if (heim_fuzzer(type, &ctx, lcount, copy, length)) {
		heim_fuzzer_free(type, ctx);
		ctx = NULL;
		break;
	    }
	}

	ret = decode_item(copy, length, data, &size);
	if (ret) {
	    memcpy(copy, p, length);
	} else {
	    free_item(data);
	}

	tcount--;
	count++;
	lcount++;
	if ((count & 0xffff) == 0) {
	    printf("%lu...\n", (unsigned long)lcount);

	    if (leak_cmd) {
		memset(data, 0, datasize);
		if (system(leak_cmd))
		    abort();
	    }
	}
    }

    map_free(copy_map, "fuzzer", "copy");
    map_free(data_map, "fuzzer", "data");
}

static void
parseTestcase(const char *filename)
{
    struct stat sb;
    char *p, *buf;
    ssize_t sret;
    size_t size;
    int fd;

    fd = open(filename, O_RDONLY, 0);
    if (fd < 0) {
	warn("failed to open: %s", filename);
	return;
    }
    if (fstat(fd, &sb) != 0)
	err(1, "failed to stat: %s", filename);
    if (!S_ISREG(sb.st_mode)) {
	close(fd);
	return;
    }

    if (sb.st_size > (off_t)(SIZE_T_MAX >> 1))
	errx(1, "%s to larger", filename);

    buf = malloc((size_t)sb.st_size);
    if (buf == NULL)
	err(1, "malloc");
    size = (size_t)sb.st_size;

    sret = read(fd, buf, size);
    if (sret < 0)
	err(1, "read");
    else if (sret != (ssize_t)size)
	errx(1, "short read");

    close(fd);

    p = memchr(buf, '\0', size);
    if (p && p != buf) {
	p++;

	runTestcase(buf, (const void *)p, size - (p - buf), NULL);
	runTestcase(buf, (const void *)p, size - (p - buf), HEIM_FUZZ_RANDOM);
	runTestcase(buf, (const void *)p, size - (p - buf), HEIM_FUZZ_BITFLIP);
	runTestcase(buf, (const void *)p, size - (p - buf), HEIM_FUZZ_BYTEFLIP);
	runTestcase(buf, (const void *)p, size - (p - buf), HEIM_FUZZ_SHORTFLIP);
	runTestcase(buf, (const void *)p, size - (p - buf), HEIM_FUZZ_WORDFLIP);
	runTestcase(buf, (const void *)p, size - (p - buf), HEIM_FUZZ_INTERESTING8);
	runTestcase(buf, (const void *)p, size - (p - buf), HEIM_FUZZ_INTERESTING16);
	runTestcase(buf, (const void *)p, size - (p - buf), HEIM_FUZZ_INTERESTING32);
#if 0
	runTestcase(buf, (const void *)p, size - (p - buf), HEIM_FUZZ_ASN1);
#endif

    } else {
	warnx("file '%s' not a valid test case", filename);
    }

    free(buf);
}

int
main(int argc, char **argv)
{
    const char *cmd;

    if (getenv("MallocStackLogging") || getenv("MallocStackLoggingNoCompact"))
	asprintf(&leak_cmd, "leaks %d > /tmp/leaks-log-pid-%d", (int)getpid(), (int)getpid());

    dso = dlopen("/usr/local/lib/libheimdal-asn1-all-templates.dylib", RTLD_LAZY);
    if (dso == NULL)
	errx(1, "dlopen: %s", dlerror());

    if (argc < 3)
	errx(1, "missing command[fuzz-random-|][file|dir] and argument");


    cmd = argv[1];

    if (strcasecmp("dir", cmd) == 0) {
	const char *dir = argv[2];
	struct dirent *de;
	DIR *d;

	d = opendir(dir);
	if (d == NULL)
	    err(1, "opendir: %s", dir);

	while ((de = readdir(d)) != NULL) {
	    char *str;
	    asprintf(&str, "%s/%.*s", dir, (int)de->d_namlen, de->d_name);
	    
	    parseTestcase(str);
	    free(str);
	}
    } else if (strcasecmp("file", cmd) == 0) {
	parseTestcase(argv[2]);
    } else {
	errx(1, "unknown command: %s", cmd);
    }

    printf("ran %lu test cases\n", count);

    return 0;
}
