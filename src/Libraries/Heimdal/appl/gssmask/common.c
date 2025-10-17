/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 26, 2022.
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
#include <common.h>
RCSID("$Id$");

krb5_error_code
store_string(krb5_storage *sp, const char *str)
{
    size_t len = strlen(str) + 1;
    krb5_error_code ret;

    ret = krb5_store_int32(sp, len);
    if (ret)
	return ret;
    ret = krb5_storage_write(sp, str, len);
    if (ret != len)
	return EINVAL;
    return 0;
}

static void
add_list(char ****list, size_t *listlen, char **str, size_t len)
{
    size_t i;
    *list = erealloc(*list, sizeof(**list) * (*listlen + 1));

    (*list)[*listlen] = ecalloc(len, sizeof(**list));
    for (i = 0; i < len; i++)
	(*list)[*listlen][i] = str[i];
    (*listlen)++;
}

static void
permute(char ****list, size_t *listlen,
	char **str, const int start, const int len)
{
    int i, j;

#define SWAP(s,i,j) { char *t = str[i]; str[i] = str[j]; str[j] = t; }

    for (i = start; i < len - 1; i++) {
	for (j = i+1; j < len; j++) {
	    SWAP(str,i,j);
	    permute(list, listlen, str, i+1, len);
	    SWAP(str,i,j);
	}
    }
    add_list(list, listlen, str, len);
}

char ***
permutate_all(struct getarg_strings *strings, size_t *size)
{
    char **list, ***all = NULL;
    int i;

    *size = 0;

    list = ecalloc(strings->num_strings, sizeof(*list));
    for (i = 0; i < strings->num_strings; i++)
	list[i] = strings->strings[i];

    permute(&all, size, list, 0, strings->num_strings);
    free(list);
    return all;
}
