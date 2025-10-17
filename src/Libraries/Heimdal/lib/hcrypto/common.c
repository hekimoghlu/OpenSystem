/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 19, 2023.
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
#include <config.h>

#include <errno.h>

#include <stdio.h>
#include <stdlib.h>
#include <heimbase.h>

#include <krb5-types.h>
#include <rfc2459_asn1.h>
#include <hcrypto/bn.h>

#ifdef HAVE_CDSA
#define NEED_CDSA 1
#endif

#include "common.h"

#ifdef HAVE_CDSA

static CSSM_CSP_HANDLE cspHandle;

static CSSM_VERSION vers = {2, 0 };
static const CSSM_GUID guid = { 0xFADE, 0, 0, { 1, 2, 3, 4, 5, 6, 7, 0 } };

const CSSM_DATA _hc_labelData = { 7, (void *)"noLabel" };

static void * cssmMalloc(CSSM_SIZE size, void *alloc)           { return malloc(size); }
static void   cssmFree(void *ptr, void *alloc)                  { free(ptr); }
static void * cssmRealloc(void *ptr, CSSM_SIZE size, void *alloc)       { return realloc(ptr, size); }
static void * cssmCalloc(uint32 num, CSSM_SIZE size, void *alloc)       { return calloc(num, size); }


static CSSM_API_MEMORY_FUNCS cssm_memory_funcs = {
    cssmMalloc,
    cssmFree,
    cssmRealloc,
    cssmCalloc,
    NULL
};
 
CSSM_CSP_HANDLE
_hc_get_cdsa_csphandle(void)
{
    CSSM_PVC_MODE pvcPolicy = CSSM_PVC_NONE;
    CSSM_RETURN ret;

    if (cspHandle)
        return cspHandle;
        
    ret = CSSM_Init(&vers, CSSM_PRIVILEGE_SCOPE_NONE,
                    &guid, CSSM_KEY_HIERARCHY_NONE,
                    &pvcPolicy, NULL);
    if (ret != CSSM_OK) {
	heim_abort("CSSM_Init failed with: %d", ret);
    }

    ret = CSSM_ModuleLoad(&gGuidAppleCSP, CSSM_KEY_HIERARCHY_NONE, NULL, NULL);
    if (ret)
        abort();

    ret = CSSM_ModuleAttach(&gGuidAppleCSP, &vers, &cssm_memory_funcs,
                            0, CSSM_SERVICE_CSP, 0,
                            CSSM_KEY_HIERARCHY_NONE,
                            NULL, 0, NULL, &cspHandle);
    if (ret) {
	heim_abort("CSSM_ModuleAttach failed with: %d", ret);
    }

    return cspHandle;
}
#endif


int
_hc_BN_to_integer(BIGNUM *bn, heim_integer *integer)
{
    integer->length = BN_num_bytes(bn);
    integer->data = malloc(integer->length);
    if (integer->data == NULL)
	return ENOMEM;
    BN_bn2bin(bn, integer->data);
    integer->negative = BN_is_negative(bn);
    return 0;
}

BIGNUM *
_hc_integer_to_BN(const heim_integer *i, BIGNUM *bn)
{
    bn = BN_bin2bn(i->data, i->length, bn);
    if (bn)
	BN_set_negative(bn, i->negative);
    return bn;
}
