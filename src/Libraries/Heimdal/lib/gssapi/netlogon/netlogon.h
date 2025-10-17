/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 8, 2023.
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
#ifndef NETLOGON_NETLOGON_H
#define NETLOGON_NETLOGON_H

#include <config.h>

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <errno.h>

#include <gssapi.h>
#include <gssapi_mech.h>
#include <gssapi_netlogon.h>

#include <krb5.h>

#include <roken.h>
#include <heim_threads.h>

#define HC_DEPRECATED_CRYPTO
#include "crypto-headers.h"

/*
 *
 */

typedef struct {
#define NL_NEGOTIATE_REQUEST_MESSAGE    0x00000000
#define NL_NEGOTIATE_RESPONSE_MESSAGE   0x00000001
    uint32_t MessageType;
#define NL_FLAG_NETBIOS_DOMAIN_NAME     0x00000001
#define NL_FLAG_NETBIOS_COMPUTER_NAME   0x00000002
#define NL_FLAG_DNS_DOMAIN_NAME         0x00000004
#define NL_FLAG_DNS_HOST_NAME           0x00000008 /* not used */
#define NL_FLAG_UTF8_COMPUTER_NAME      0x00000010
    uint32_t Flags;
    char *Buffer[];
} NL_AUTH_MESSAGE;

#define NL_AUTH_MESSAGE_LENGTH          8

/* SignatureAlgorithm */
#define NL_SIGN_ALG_HMAC_MD5            0x0077
#define NL_SIGN_ALG_SHA256              0x0013

/* SealAlgorithm */
#define NL_SEAL_ALG_RC4                 0x007A
#define NL_SEAL_ALG_AES128              0x001A
#define NL_SEAL_ALG_NONE                0xFFFF

typedef struct {
    uint16_t SignatureAlgorithm;
    uint16_t SealAlgorithm;
    uint16_t Pad;
    uint16_t Flags;
    uint8_t SequenceNumber[8];
    uint8_t Checksum[8];
    uint8_t Confounder[8];
} NL_AUTH_SIGNATURE;

#define NL_AUTH_SIGNATURE_HEADER_LENGTH 8
#define NL_AUTH_SIGNATURE_COMMON_LENGTH 16
#define NL_AUTH_SIGNATURE_LENGTH        32

typedef struct {
    uint16_t SignatureAlgorithm;
    uint16_t SealAlgorithm;
    uint16_t Pad;
    uint16_t Flags;
    uint8_t SequenceNumber[8];
    uint8_t Checksum[32];
    uint8_t Confounder[8];
} NL_AUTH_SHA2_SIGNATURE;

#define NL_AUTH_SHA2_SIGNATURE_LENGTH   56

typedef union {
    NL_AUTH_SIGNATURE Signature;
    NL_AUTH_SHA2_SIGNATURE SHA2Signature;
} NL_AUTH_SIGNATURE_U;

#define NL_AUTH_SIGNATURE_P(_u)         (&(_u)->Signature)

typedef struct gssnetlogon_name {
    gss_buffer_desc NetbiosName;
    gss_buffer_desc DnsName;
} *gssnetlogon_name;

typedef struct gssnetlogon_cred {
    gssnetlogon_name *Name;
    uint16_t SignatureAlgorithm;
    uint16_t SealAlgorithm;
    uint8_t SessionKey[16];
} *gssnetlogon_cred;

typedef struct gssnetlogon_ctx {
    HEIMDAL_MUTEX Mutex;
    enum { NL_AUTH_NEGOTIATE, NL_AUTH_ESTABLISHED } State;
    OM_uint32 GssFlags;
    uint8_t LocallyInitiated;
    uint32_t MessageBlockSize;
    uint16_t SignatureAlgorithm;
    uint16_t SealAlgorithm;
    uint64_t SequenceNumber;
    gssnetlogon_name SourceName;
    gssnetlogon_name TargetName;
    uint8_t SessionKey[16];
} *gssnetlogon_ctx;

#include <netlogon-private.h>

#endif /* NETLOGON_NETLOGON_H */
