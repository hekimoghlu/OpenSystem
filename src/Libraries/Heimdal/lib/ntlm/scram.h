/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 13, 2024.
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
#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include <krb5-types.h>

#include <wind.h>
#include <roken.h>
#include <base64.h>

#include <heimbase.h>

#include "heimscram.h"

#include <ntlm_err.h>
#include "crypto-headers.h"

#ifndef __APPLE_TARGET_EMBEDDED__
#include <CommonCrypto/CommonKeyDerivation.h>
#endif

struct heim_scram_pair {
    char type;
    heim_scram_data data;
};

struct heim_scram_pairs {
    int flags;
#define SCRAM_PAIR_ALLOCATED 1
#define SCRAM_ARRAY_ALLOCATED 2
#define SCRAM_BINDINGS_YES 4
#define SCRAM_BINDINGS_NO 8
    struct heim_scram_pair *val;
    size_t len;
};

typedef struct heim_scram_pairs heim_scram_pairs;

struct heim_scram {
    struct heim_scram_method_desc *method;
    enum { CLIENT, SERVER } type;
    heim_scram_data client1;
    heim_scram_data server1;
    /* generated */
    heim_scram_data nonce;

    /* server */
    struct heim_scram_server *server;
    void *ctx;

    heim_scram_data user;

    /* output */
    heim_scram_data ClientProof;
    heim_scram_data ServerSignature;
    heim_scram_data SessionKey;
};

#include "heimscram-protos.h"

int
_heim_scram_parse(heim_scram_data *data, heim_scram_pairs **pd);

int
_heim_scram_unparse (
	heim_scram_pairs */*d*/,
	heim_scram_data */*out*/);

void
_heim_scram_pairs_free (heim_scram_pairs */*d*/);
