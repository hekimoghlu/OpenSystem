/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 3, 2024.
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
#include "config.h"

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "dav1d_fuzzer.h"

// expects ivf input

int main(int argc, char *argv[]) {
    int ret = -1;
    FILE *f = NULL;
    int64_t fsize;
    const char *filename = NULL;
    uint8_t *data = NULL;
    size_t size = 0;

    if (LLVMFuzzerInitialize(&argc, &argv)) {
        return 1;
    }

    if (argc != 2) {
        fprintf(stdout, "Usage:\n%s fuzzing_testcase.ivf\n", argv[0]);
        return -1;
    }
    filename = argv[1];

    if (!(f = fopen(filename, "rb"))) {
        fprintf(stderr, "failed to open %s: %s\n", filename, strerror(errno));
        goto error;
    }

    if (fseeko(f, 0, SEEK_END) == -1) {
        fprintf(stderr, "fseek(%s, 0, SEEK_END) failed: %s\n", filename,
                strerror(errno));
        goto error;
    }
    if ((fsize = ftello(f)) == -1) {
        fprintf(stderr, "ftell(%s) failed: %s\n", filename, strerror(errno));
        goto error;
    }
    rewind(f);

    if (fsize < 0 || fsize > INT_MAX) {
        fprintf(stderr, "%s is too large: %"PRId64"\n", filename, fsize);
        goto error;
    }
    size = (size_t)fsize;

    if (!(data = malloc(size))) {
        fprintf(stderr, "failed to allocate: %zu bytes\n", size);
        goto error;
    }

    if (fread(data, size, 1, f) == size) {
        fprintf(stderr, "failed to read %zu bytes from %s: %s\n", size,
                filename, strerror(errno));
        goto error;
    }

    ret = LLVMFuzzerTestOneInput(data, size);

error:
    free(data);
    if (f) fclose(f);
    return ret;
}
