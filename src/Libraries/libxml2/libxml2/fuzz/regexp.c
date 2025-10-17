/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 14, 2024.
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
#include <libxml/xmlregexp.h>
#include "fuzz.h"

int
LLVMFuzzerInitialize(int *argc ATTRIBUTE_UNUSED,
                     char ***argv ATTRIBUTE_UNUSED) {
    xmlSetGenericErrorFunc(NULL, xmlFuzzErrorFunc);

    return 0;
}

int
LLVMFuzzerTestOneInput(const char *data, size_t size) {
    xmlRegexpPtr regexp;
    char *str[2] = { NULL, NULL };
    size_t numStrings;

    if (size > 200)
        return(0);

    numStrings = xmlFuzzExtractStrings(data, size, str, 2);

    /* CUR_SCHAR doesn't handle invalid UTF-8 and may cause infinite loops. */
    if (xmlCheckUTF8(BAD_CAST str[0]) != 0) {
        regexp = xmlRegexpCompile(BAD_CAST str[0]);
        {
            FILE *fd = fopen("/dev/null", "w");
            xmlRegexpPrint(fd, regexp);
            fclose(fd);
        }
        /* xmlRegexpExec has pathological performance in too many cases. */
#if 0
        if ((regexp != NULL) && (numStrings >= 2)) {
            xmlRegexpExec(regexp, BAD_CAST str[1]);
        }
#endif
        xmlRegFreeRegexp(regexp);
    }

    xmlFree(str[0]);
    xmlFree(str[1]);
    xmlResetLastError();

    return 0;
}

