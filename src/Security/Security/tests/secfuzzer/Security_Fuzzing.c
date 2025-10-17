/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 15, 2025.
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

//
//  Security_Fuzzing.c
//  Security
//

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "SecCertificateFuzzer.h"
#include "SecKeychainFuzzer.h"

#define SEC_FUZZER_MODE_ENV_VAR "SEC_FUZZER_MODE"

#define MODESTR_SECCERTIFICATE "SecCertificate"
#define MODESTR_SECKEYCHAIN    "SecKeychain"

int LLVMFuzzerInitialize(int *argc, char ***argv);

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t len);

typedef enum {
    MODE_UNKNOWN = 0,
    MODE_SECCERTIFICATE,
    MODE_SECKEYCHAIN
} mode_enum;

static mode_enum mode = MODE_UNKNOWN;

int LLVMFuzzerInitialize(int *argc, char ***argv) {
    char* mode_env = getenv(SEC_FUZZER_MODE_ENV_VAR);

    if (!mode_env || 0 == strncmp(mode_env, MODESTR_SECCERTIFICATE, strlen(MODESTR_SECCERTIFICATE))) {
        mode = MODE_SECCERTIFICATE;
    } else if (0 == strncmp(mode_env, MODESTR_SECKEYCHAIN, strlen(MODESTR_SECKEYCHAIN))) {
        mode = MODE_SECKEYCHAIN;
    }

    if (mode == MODE_UNKNOWN) {
        printf("Unknown mode (from env var %s): %s\n", SEC_FUZZER_MODE_ENV_VAR, mode_env);
        exit(1);
    }

    return 0;
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t len)
{
    if (mode == MODE_SECCERTIFICATE) {
        SecCertificateFuzzer(data, len);
    } else if (mode == MODE_SECKEYCHAIN) {
        SecKeychainFuzzer(data, len);
    }

    return 0;
}
