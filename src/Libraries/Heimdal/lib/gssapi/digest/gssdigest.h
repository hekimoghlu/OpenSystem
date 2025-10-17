/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 8, 2023.
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

#ifndef NTLM_NTLM_H
#define NTLM_NTLM_H

#include <config.h>

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <errno.h>

#include <roken.h>

#include <gssapi.h>
#include <gssapi_scram.h>
#include <gssapi_mech.h>
#include <gssapi_spi.h>

#include <krb5.h>
#include <heim_threads.h>

#include <kcm.h>

#include <gssdigest.h>
#include <heimscram.h>

#define HC_DEPRECATED_CRYPTO
#include "crypto-headers.h"

typedef struct {
    char *name;
    unsigned char uuid[16];
} scram_c_desc, *scram_c;

typedef scram_c scram_cred;

typedef struct {
#ifdef HAVE_KCM
    char *client;
#else
    scram_cred client;
#endif
    heim_scram *scram;

    OM_uint32 flags;
    uint32_t status;
#define STATUS_OPEN 1
#define STATUS_CLIENT 2
#define STATUS_SESSIONKEY 4

    int state;
#define CLIENT1		1
#define SERVER1		2
#define CLIENT2		3
#define SERVER2		4
#define CLIENT3		5

    heim_scram_data sessionkey;

} *scram_id_t;

#include <digest-private.h>


#endif /* NTLM_NTLM_H */
