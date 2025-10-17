/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 1, 2021.
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
#include "krb5_locl.h"

#ifndef HEIMDAL_SMALLER
#define DES3_OLD_ENCTYPE 1
#endif

static struct _krb5_key_type keytype_null = {
    KRB5_ENCTYPE_NULL,
    "null",
    0,
    0,
    0,
    NULL,
    NULL,
    NULL
};

static krb5_error_code
NONE_checksum(krb5_context context,
	      struct _krb5_key_data *key,
	      const void *data,
	      size_t len,
	      unsigned usage,
	      Checksum *C)
{
    return 0;
}

struct _krb5_checksum_type _krb5_checksum_none = {
    CKSUMTYPE_NONE,
    "none",
    1,
    0,
    0,
    NONE_checksum,
    NULL
};

static krb5_error_code
NULL_encrypt(krb5_context context,
	     struct _krb5_key_data *key,
	     void *data,
	     size_t len,
	     krb5_boolean encryptp,
	     int usage,
	     void *ivec)
{
    return 0;
}

struct _krb5_encryption_type _krb5_enctype_null = {
    ETYPE_NULL,
    "null",
    1,
    1,
    0,
    &keytype_null,
    &_krb5_checksum_none,
    NULL,
    F_DISABLED,
    NULL_encrypt,
    0,
    NULL
};
