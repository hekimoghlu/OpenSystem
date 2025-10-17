/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 8, 2024.
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
#include <stdio.h>
#include <stdlib.h>

int
main(int argc, char *argv[])
{
    FILE *bfile;
    int ch;

    if (argc != 2) {
        fprintf(stderr, "Usage: %s <file>\n", argv[0]);
        return EXIT_FAILURE;
    }

    bfile = fopen(argv[1], "rb");

    if (!bfile) {
        fprintf(stderr, "Could not open `%s'.\n", argv[1]);
        return EXIT_FAILURE;
    }

    while ((ch = fgetc(bfile)) != EOF)
        printf("%02x \n", ch);

    if (ferror(bfile)) {
        fprintf(stderr, "Error reading from `%s'.\n", argv[1]);
        return EXIT_FAILURE;
    }

    fclose(bfile);
    return EXIT_SUCCESS;
}
