/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 30, 2024.
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
#include <err.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#include "../disklib/fskit_support.h"


#define DEFAULT_EXIT_CODE 8 /* Standard Error exit code for fsck */

static void usage();

static void usage(void) {
    fprintf(stderr, "Usage: fsck_fskit [-t fstype] <options> device\n");
    exit(DEFAULT_EXIT_CODE);
}

int main(int argc, const char * argv[]) {

    int ret;
    bool typeOptionFound = false;
    bool typeValueFound = false;

    if (argc < 4) {
        usage();
    }

    // We are only supporting `fsck_fskit -t fstype <otheroptions> someDisk`, different filesystem use different options, so we are going
    // to leave the option parsing to FSKit.

    if (strcmp(argv[1], "-t") == 0) {
        typeOptionFound = true;
    }
    if (strncmp("-", argv[2], 1) != 0) {
        typeValueFound = true;
    }

    if(!typeOptionFound || !typeValueFound) {
        errx(1, "No file system type was provided");
    }

    // invoke_tool_from_fskit expects arguments to look like "fstype <other_options> disk", so we are ignoring `fsck_fskit` and `-t` arguments.
    argc -= 2;
    argv += 2;

    ret = invoke_tool_from_fskit(check_fs_op, 0, argc, argv);
    exit(ret);

}
