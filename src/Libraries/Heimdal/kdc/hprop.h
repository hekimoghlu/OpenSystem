/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 12, 2023.
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

#ifndef __HPROP_H__
#define __HPROP_H__

#include "headers.h"

struct prop_data{
    krb5_context context;
    krb5_auth_context auth_context;
    int sock;
};

#define HPROP_VERSION "hprop-0.0"
#define HPROP_NAME "hprop"
#define HPROP_KEYTAB "HDB:"
#define HPROP_PORT 754

#ifndef NEVERDATE
#define NEVERDATE ((1U << 31) - 1)
#endif

struct v4_principal {
    char name[64];
    char instance[64];
    DES_cblock key;
    int kvno;
    int mkvno;
    time_t exp_date;
    time_t mod_date;
    char mod_name[64];
    char mod_instance[64];
    int max_life;
};

int v4_prop(void*, struct v4_principal*);
int v4_prop_dump(void *arg, const char*);

#endif /* __HPROP_H__ */
