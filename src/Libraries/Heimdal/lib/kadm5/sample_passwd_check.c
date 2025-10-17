/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 11, 2022.
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
/* $Id$ */

#include <string.h>
#include <stdlib.h>
#include <krb5.h>

const char* check_length(krb5_context, krb5_principal, krb5_data *);

/* specify the api-version this library conforms to */

int version = 0;

/* just check the length of the password, this is what the default
   check does, but this lets you specify the minimum length in
   krb5.conf */
const char*
check_length(krb5_context context,
             krb5_principal prinipal,
             krb5_data *password)
{
    int min_length = krb5_config_get_int_default(context, NULL, 6,
						 "password_quality",
						 "min_length",
						 NULL);
    if(password->length < min_length)
	return "Password too short";
    return NULL;
}

#ifdef DICTPATH

/* use cracklib to check password quality; this requires a patch for
   cracklib that can be found at
   ftp://ftp.pdc.kth.se/pub/krb/src/cracklib.patch */

const char*
check_cracklib(krb5_context context,
	       krb5_principal principal,
	       krb5_data *password)
{
    char *s = malloc(password->length + 1);
    char *msg;
    char *strings[2];
    if(s == NULL)
	return NULL; /* XXX */
    strings[0] = principal->name.name_string.val[0]; /* XXX */
    strings[1] = NULL;
    memcpy(s, password->data, password->length);
    s[password->length] = '\0';
    msg = FascistCheck(s, DICTPATH, strings);
    memset(s, 0, password->length);
    free(s);
    return msg;
}
#endif
