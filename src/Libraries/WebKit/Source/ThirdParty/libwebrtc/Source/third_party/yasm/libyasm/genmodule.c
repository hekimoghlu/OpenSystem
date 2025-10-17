/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 15, 2024.
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
#include <string.h>
#include <ctype.h>

#include "compat-queue.h"

#define OUTPUT  "module.c"
#define MAXNAME 128
#define MAXLINE 1024
#define MAXMODULES 128
#define MAXINCLUDES 256

typedef struct include {
    STAILQ_ENTRY(include) link;
    char *filename;
} include;

int
main(int argc, char *argv[])
{
    FILE *in, *out;
    char *str;
    int i;
    size_t len;
    char *strp;
    char *modules[MAXMODULES];
    int num_modules = 0;
    STAILQ_HEAD(includehead, include) includes =
        STAILQ_HEAD_INITIALIZER(includes);
    include *inc;
    int isam = 0;
    int linecont = 0;

    if (argc != 3) {
        fprintf(stderr, "Usage: %s <module.in> <Makefile[.am]>\n", argv[0]);
        return EXIT_FAILURE;
    }

    str = malloc(MAXLINE);

    /* Starting with initial input Makefile, look for include <file> or
     * YASM_MODULES += <module>.  Note this currently doesn't handle
     * a relative starting path.
     */
    len = strlen(argv[2]);
    inc = malloc(sizeof(include));
    inc->filename = malloc(len+1);
    strcpy(inc->filename, argv[2]);
    STAILQ_INSERT_TAIL(&includes, inc, link);

    isam = argv[2][len-2] == 'a' && argv[2][len-1] == 'm';

    while (!STAILQ_EMPTY(&includes)) {
        inc = STAILQ_FIRST(&includes);
        STAILQ_REMOVE_HEAD(&includes, link);
        in = fopen(inc->filename, "rt");
        if (!in) {
            fprintf(stderr, "Could not open `%s'.\n", inc->filename);
            return EXIT_FAILURE;
        }
        free(inc->filename);
        free(inc);

        while (fgets(str, MAXLINE, in)) {
            /* Strip off any trailing whitespace */
            len = strlen(str);
            if (len > 0) {
                strp = &str[len-1];
                while (len > 0 && isspace(*strp)) {
                    *strp-- = '\0';
                    len--;
                }
            }

            strp = str;

            /* Skip whitespace */
            while (isspace(*strp))
                strp++;

            /* Skip comments */
            if (*strp == '#')
                continue;

            /* If line continuation, skip to continue copy */
            if (linecont)
                goto keepgoing;

            /* Check for include if original input is .am file */
            if (isam && strncmp(strp, "include", 7) == 0 && isspace(strp[7])) {
                strp += 7;
                while (isspace(*strp))
                    strp++;
                /* Build new include and add to end of list */
                inc = malloc(sizeof(include));
                inc->filename = malloc(strlen(strp)+1);
                strcpy(inc->filename, strp);
                STAILQ_INSERT_TAIL(&includes, inc, link);
                continue;
            }

            /* Check for YASM_MODULES = or += */
            if (strncmp(strp, "YASM_MODULES", 12) != 0)
                continue;
            strp += 12;
            while (isspace(*strp))
                strp++;
            if (strncmp(strp, "+=", 2) != 0 && *strp != '=')
                continue;
            if (*strp == '+')
                strp++;
            strp++;
            while (isspace(*strp))
                strp++;

keepgoing:
            /* Check for continuation */
            if (len > 0 && str[len-1] == '\\') {
                str[len-1] = '\0';
                while (isspace(*strp))
                    *strp-- = '\0';
                linecont = 1;
            } else
                linecont = 0;

            while (*strp != '\0') {
                /* Copy module name */
                modules[num_modules] = malloc(MAXNAME);
                len = 0;
                while (*strp != '\0' && !isspace(*strp))
                    modules[num_modules][len++] = *strp++;
                modules[num_modules][len] = '\0';
                num_modules++;

                while (isspace(*strp))
                    strp++;
            }
        }
        fclose(in);
    }

    out = fopen(OUTPUT, "wt");

    if (!out) {
        fprintf(stderr, "Could not open `%s'.\n", OUTPUT);
        return EXIT_FAILURE;
    }

    fprintf(out, "/* This file auto-generated by genmodule.c"
                 " - don't edit it */\n\n");

    in = fopen(argv[1], "rt");
    if (!in) {
        fprintf(stderr, "Could not open `%s'.\n", argv[1]);
        fclose(out);
        remove(OUTPUT);
        return EXIT_FAILURE;
    }

    len = 0;
    while (fgets(str, MAXLINE, in)) {
        if (strncmp(str, "MODULES_", 8) == 0) {
            len = 0;
            strp = str+8;
            while (*strp != '\0' && *strp != '_') {
                len++;
                strp++;
            }
            *strp = '\0';

            for (i=0; i<num_modules; i++) {
                if (strncmp(modules[i], str+8, len) == 0) {
                    fprintf(out, "    {\"%s\", &yasm_%s_LTX_%s},\n",
                            modules[i]+len+1, modules[i]+len+1, str+8);
                }
            }
        } else if (strncmp(str, "EXTERN_LIST", 11) == 0) {
            for (i=0; i<num_modules; i++) {
                strcpy(str, modules[i]);
                strp = str;
                while (*strp != '\0' && *strp != '_')
                    strp++;
                *strp++ = '\0';

                fprintf(out, "extern yasm_%s_module yasm_%s_LTX_%s;\n",
                        str, strp, str);
            }
        } else
            fputs(str, out);
    }

    fclose(in);
    fclose(out);

    for (i=0; i<num_modules; i++)
        free(modules[i]);
    free(str);

    return EXIT_SUCCESS;
}
