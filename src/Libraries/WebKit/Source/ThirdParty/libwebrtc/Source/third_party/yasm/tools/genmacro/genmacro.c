/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 19, 2022.
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

#define MAXLINE 1024

int
main(int argc, char *argv[])
{
    FILE *in, *out;
    int i;
    char *str;
    char *strp;
    char *charp;
    int fline;
    int line = 0;
    int lindex = 0;
    size_t len;

    if (argc < 4) {
        fprintf(stderr, "Usage: %s <out> <var> <file> [<file> ...]\n", argv[0]);
        return EXIT_FAILURE;
    }

    out = fopen(argv[1], "wt");

    if (!out) {
        fprintf(stderr, "Could not open `%s'.\n", argv[1]);
        return EXIT_FAILURE;
    }

    str = malloc(MAXLINE);

    fprintf(out, "/* This file auto-generated from standard.mac by genmacro.c"
                 " - don't edit it */\n\n#include <stddef.h>\n\n"
                 "static const char *%s[] = {\n", argv[2]);

    for (i=3; i<argc; i++) {
        in = fopen(argv[i], "rt");
        if (!in) {
            fprintf(stderr, "Could not open `%s'.\n", argv[i]);
            fclose(out);
            remove(argv[1]);
            return EXIT_FAILURE;
        }

        fline = 0;

        while (fgets(str, MAXLINE, in)) {
            line++;
            fline++;

            strp = str;

            /* check for unterminated quotes and delete comments */
            charp = strp;
            while ((charp = strpbrk(charp, "'\";"))) {
                if (charp[0] == ';') {
                    *charp = '\0';
                    break;
                }
                if ((charp = strchr(charp+1, charp[0])) == NULL) {
                    fprintf(stderr, "%s:%d: error: unterminated quote\n",
                            argv[i], fline);
                    fclose(out);
                    remove(argv[1]);
                    return EXIT_FAILURE;
                }
                charp++;
            }

            /* strip off leading and trailing whitespace */
            while (*strp == ' ' || *strp == '\t')
                strp++;
            len = strlen(strp);
            while (len > 0 && (strp[len-1] == ' ' || strp[len-1] == '\t' ||
                               strp[len-1] == '\n')) {
                strp[len-1] = '\0';
                len--;
            }

            /* skip blank lines */
            if (len == 0)
                continue;

            /* output as string to output file */
            fprintf(out, "    \"");
            while (*strp != '\0') {
                if (*strp == '\\' || *strp == '"')
                    fputc('\\', out);
                fputc(*strp, out);
                strp++;
            }
            fprintf(out, "\",\n");
            lindex++;
        }

        fclose(in);
    }

    fprintf(out, "    NULL\n};\n");
    fclose(out);

    free(str);

    return EXIT_SUCCESS;
}
