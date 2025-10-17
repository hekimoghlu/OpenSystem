/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 10, 2024.
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
#include <sysexits.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "pktmetadatafilter.h"

int main(int argc,  char * const argv[])
{
    int ch;
    char *input_str = NULL;
    node_t *pkt_meta_data_expression = NULL;
    int verbose = 0;

    while ((ch = getopt(argc, argv, "Q:v")) != -1) {
        switch (ch) {
            case 'Q':
                if (input_str != NULL) {
                    errx(EX_USAGE, "-Q used twice");
                }
                input_str = strdup(optarg);
                if (input_str == NULL) {
                    errx(EX_OSERR, "calloc() failed");
                }
                break;
            case 'v':
                verbose = 1;
                break;
        }
    }
    set_parse_verbose(verbose);

    pkt_meta_data_expression = parse_expression(input_str);
    if (pkt_meta_data_expression == NULL)
        errx(EX_SOFTWARE, "invalid expression \"%s\"", input_str);

    free(input_str);

    return 0;
}
