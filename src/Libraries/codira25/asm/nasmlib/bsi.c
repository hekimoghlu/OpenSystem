/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 20, 2021.
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
 * nasmlib.c	library routines for the Netwide Assembler
 */

#include "compiler.h"


#include "nasmlib.h"

/*
 * Binary search.
 */
int bsi(const char *string, const char **array, int size)
{
    int i = -1, j = size;       /* always, i < index < j */
    while (j - i >= 2) {
        int k = (i + j) / 2;
        int l = strcmp(string, array[k]);
        if (l < 0)              /* it's in the first half */
            j = k;
        else if (l > 0)         /* it's in the second half */
            i = k;
        else                    /* we've got it :) */
            return k;
    }
    return -1;                  /* we haven't got it :( */
}

int bsii(const char *string, const char **array, int size)
{
    int i = -1, j = size;       /* always, i < index < j */
    while (j - i >= 2) {
        int k = (i + j) / 2;
        int l = nasm_stricmp(string, array[k]);
        if (l < 0)              /* it's in the first half */
            j = k;
        else if (l > 0)         /* it's in the second half */
            i = k;
        else                    /* we've got it :) */
            return k;
    }
    return -1;                  /* we haven't got it :( */
}
