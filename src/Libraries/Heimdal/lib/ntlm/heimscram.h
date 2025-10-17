/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 14, 2025.
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
#ifndef HEIM_SCRAM_H
#define HEIM_SCRAM_H

#include <sys/types.h>

#ifndef __HEIM_BASE_DATA__
#define __HEIM_BASE_DATA__ 1
struct heim_base_data {
	size_t length;
	void *data;
};
#endif
typedef struct heim_base_data heim_scram_data;

typedef struct heim_scram heim_scram;

typedef struct heim_scram_method_desc *heim_scram_method;

extern struct heim_scram_method_desc heim_scram_digest_sha1_s;
extern struct heim_scram_method_desc heim_scram_digest_sha256_s;

#define HEIM_SCRAM_DIGEST_SHA1 (&heim_scram_digest_sha1_s)
#define HEIM_SCRAM_DIGEST_SHA256 (&heim_scram_digest_sha256_s)

struct heim_scram_server {
#define SCRAM_SERVER_VERSION_1 1
    int version;
    int (*param)(void *ctx,
		 const heim_scram_data *user,
		 heim_scram_data *salt,
		 unsigned int *iteration,
		 heim_scram_data *servernonce);
    int (*calculate)(void *ctx,
		     heim_scram_method method,
		     const heim_scram_data *user,
		     const heim_scram_data *c1,
		     const heim_scram_data *s1,
		     const heim_scram_data *c2noproof,
		     const heim_scram_data *proof,
		     heim_scram_data *server,
		     heim_scram_data *sessionKey);
};

struct heim_scram_client {
#define SCRAM_CLIENT_VERSION_1 1
    int version;
    int (*calculate)(void *ctx,
		     heim_scram_method method,
		     unsigned int iterations,
		     heim_scram_data *salt,
		     const heim_scram_data *c1,
		     const heim_scram_data *s1,
		     const heim_scram_data *c2noproof,
		     heim_scram_data *proof,
		     heim_scram_data *server,
		     heim_scram_data *sessionKey);
};

extern struct heim_scram_client heim_scram_client_password_procs_s;
#define HEIM_SCRAM_CLIENT_PASSWORD_PROCS (&heim_scram_client_password_procs_s)

#include <heimscram-protos.h>

#endif /* SCRAM_SCRAM_H */
