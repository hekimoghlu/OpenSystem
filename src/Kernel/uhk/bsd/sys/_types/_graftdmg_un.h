/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 25, 2024.
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
#ifndef _GRAFTDMG_UN_
#define _GRAFTDMG_UN_

#include <sys/_types/_u_int8_t.h>
#include <sys/_types/_u_int64_t.h>
#include <sys/_types/_u_int32_t.h>

#define GRAFTDMG_SECURE_BOOT_CRYPTEX_ARGS_VERSION 1
#define MAX_GRAFT_ARGS_SIZE 512

/* Flag values for secure_boot_cryptex_args.sbc_flags */
#define SBC_PRESERVE_MOUNT              0x0001  /* Preserve underlying mount until shutdown */
#define SBC_ALTERNATE_SHARED_REGION     0x0002  /* Binaries within should use alternate shared region */
#define SBC_SYSTEM_CONTENT              0x0004  /* Cryptex contains system content */
#define SBC_PANIC_ON_AUTHFAIL           0x0008  /* On failure to authenticate, panic */
#define SBC_STRICT_AUTH                 0x0010  /* Strict authentication mode */
#define SBC_PRESERVE_GRAFT              0x0020  /* Preserve graft itself until unmount */

typedef struct secure_boot_cryptex_args {
	u_int32_t sbc_version;
	u_int32_t sbc_4cc;
	int sbc_authentic_manifest_fd;
	int sbc_user_manifest_fd;
	int sbc_payload_fd;
	u_int64_t sbc_flags;
} __attribute__((aligned(4), packed))  secure_boot_cryptex_args_t;

typedef union graft_args {
	u_int8_t max_size[MAX_GRAFT_ARGS_SIZE];
	secure_boot_cryptex_args_t sbc_args;
} graftdmg_args_un;

#endif /* _GRAFTDMG_UN_ */
