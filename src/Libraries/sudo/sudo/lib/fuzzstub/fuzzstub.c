/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 5, 2023.
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

#include <sys/stat.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <limits.h>
#include <fcntl.h>
#include <time.h>
#include <unistd.h>
#if defined(HAVE_STDINT_H)
# include <stdint.h>
#elif defined(HAVE_INTTYPES_H)
# include <inttypes.h>
#endif

#include "sudo_compat.h"
#include "sudo_util.h"

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);

sudo_dso_public int main(int argc, char *argv[]);

/*
 * Simple driver for fuzzers built for LLVM libfuzzer.
 * This stub library allows fuzz targets to be built and run without
 * libfuzzer.  No actual fuzzing will occur but the provided inputs
 * will be tested.
 */
int
main(int argc, char *argv[])
{
    struct timespec start_time, stop_time;
    size_t filesize, bufsize = 0;
    ssize_t nread;
    struct stat sb;
    uint8_t *buf = NULL;
    int fd, i, errors = 0;
    int verbose = 0;
    long ms;

    /* Test provided input files. */
    for (i = 1; i < argc; i++) {
	const char *arg = argv[i];
	if (*arg == '-') {
	    if (strncmp(arg, "-verbosity=", sizeof("-verbosity=") - 1) == 0) {
		verbose = atoi(arg + sizeof("-verbosity=") - 1);
	    }
	    continue;
	}
	fd = open(arg, O_RDONLY);
	if (fd == -1 || fstat(fd, &sb) != 0) {
	    fprintf(stderr, "open %s: %s\n", arg, strerror(errno));
	    if (fd != -1)
		close(fd);
	    errors++;
	    continue;
	}
#ifndef __LP64__
	if (sizeof(sb.st_size) > sizeof(size_t) && sb.st_size > SSIZE_MAX) {
	    errno = E2BIG;
	    fprintf(stderr, "%s: %s\n", arg, strerror(errno));
	    close(fd);
	    errors++;
	    continue;
	}
#endif
	filesize = sb.st_size;
	if (bufsize < filesize) {
	    void *tmp = realloc(buf, filesize);
	    if (tmp == NULL) {
		fprintf(stderr, "realloc: %s\n", strerror(errno));
		close(fd);
		errors++;
		continue;
	    }
	    buf = tmp;
	    bufsize = filesize;
	}
	nread = read(fd, buf, filesize);
	if ((size_t)nread != filesize) {
	    if (nread == -1)
		fprintf(stderr, "read %s: %s\n", arg, strerror(errno));
	    else
		fprintf(stderr, "read %s: short read\n", arg);
	    close(fd);
	    errors++;
	    continue;
	}
	close(fd);

	/* NOTE: doesn't support LLVMFuzzerInitialize() (but we don't use it) */
	if (verbose > 0) {
	    fprintf(stderr, "Running: %s\n", arg);
	    sudo_gettime_mono(&start_time);
	}
	LLVMFuzzerTestOneInput(buf, nread);
	if (verbose > 0) {
	    sudo_gettime_mono(&stop_time);
	    sudo_timespecsub(&stop_time, &start_time, &stop_time);
	    ms = (stop_time.tv_sec * 1000) + (stop_time.tv_nsec / 1000000);
	    fprintf(stderr, "Executed %s in %ld ms\n", arg, ms);
	}
    }
    free(buf);

    return errors;
}
