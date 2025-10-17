/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 3, 2025.
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
#include <string.h>
#include <stdlib.h>

#include <libxslt/transform.h>
#include <libxslt/xsltutils.h>

extern int xmlLoadExtDtdDefaultValue;

static void usage(const char *name) {
    printf("Usage: %s [options] stylesheet [stylesheet ...] file [file ...]\n",
            name);
    printf("      --out file: send output to file\n");
    printf("      --param name value: pass a (parameter,value) pair\n");
}

int main(int argc, char **argv) {
    int arg_indx;
    const char *params[16 + 1];
    int params_indx = 0;
    int stylesheet_indx = 0;
    int file_indx = 0;
    int i, j, k;
    FILE *output_file = stdout;
    xsltStylesheetPtr *stylesheets = 
        (xsltStylesheetPtr *) calloc(argc, sizeof(xsltStylesheetPtr));
    xmlDocPtr *files = (xmlDocPtr *) calloc(argc, sizeof(xmlDocPtr));
    xmlDocPtr doc, res;
    int return_value = 0;
        
    if (argc <= 1) {
        usage(argv[0]);
        return_value = 1;
        goto finish;
    }
        
    /* Collect arguments */
    for (arg_indx = 1; arg_indx < argc; arg_indx++) {
        if (argv[arg_indx][0] != '-')
            break;
        if ((!strcmp(argv[arg_indx], "-param"))
                || (!strcmp(argv[arg_indx], "--param"))) {
            arg_indx++;
            params[params_indx++] = argv[arg_indx++];
            params[params_indx++] = argv[arg_indx];
            if (params_indx >= 16) {
                fprintf(stderr, "too many params\n");
                return_value = 1;
                goto finish;
            }
        }  else if ((!strcmp(argv[arg_indx], "-o"))
                || (!strcmp(argv[arg_indx], "--out"))) {
            arg_indx++;
            output_file = fopen(argv[arg_indx], "w");
        } else {
            fprintf(stderr, "Unknown option %s\n", argv[arg_indx]);
            usage(argv[0]);
            return_value = 1;
            goto finish;
        }
    }
    params[params_indx] = 0;

    /* Collect and parse stylesheets and files to be transformed */
    for (; arg_indx < argc; arg_indx++) {
        size_t argument_length = sizeof(char) * (strlen(argv[arg_indx]) + 1);
        char *argument = (char *) malloc(argument_length);
        if (!argument) {
            /* Out of memory */
            exit(1);
        }
        strncpy(argument, argv[arg_indx], argument_length);
        if (strtok(argument, ".")) {
            char *suffix = strtok(0, ".");
            if (suffix && !strcmp(suffix, "xsl")) {
                stylesheets[stylesheet_indx++] =
                    xsltParseStylesheetFile((const xmlChar *)argv[arg_indx]);;
            } else {
                files[file_indx++] = xmlParseFile(argv[arg_indx]);
            }
        } else {
            files[file_indx++] = xmlParseFile(argv[arg_indx]);
        }
        free(argument);
    }

    xmlSubstituteEntitiesDefault(1);
    xmlLoadExtDtdDefaultValue = 1;

    /* Process files */
    for (i = 0; files[i]; i++) {
        doc = files[i];
        res = doc;
        for (j = 0; stylesheets[j]; j++) {
            res = xsltApplyStylesheet(stylesheets[j], doc, params);
            xmlFreeDoc(doc);
            doc = res;
        }

        if (stylesheets[0]) {
            xsltSaveResultToFile(output_file, res, stylesheets[j-1]);
        } else {
            xmlDocDump(output_file, res);
        }
        xmlFreeDoc(res);
    }

    fclose(output_file);

    for (k = 0; stylesheets[k]; k++) {
        xsltFreeStylesheet(stylesheets[k]);
    }

    xsltCleanupGlobals();
    xmlCleanupParser();

 finish:
    free(stylesheets);
    free(files);
    return(return_value);
}
