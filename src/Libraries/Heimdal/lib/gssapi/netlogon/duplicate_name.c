/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 27, 2022.
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
#include "netlogon.h"

OM_uint32 _netlogon_duplicate_name (
            OM_uint32 * minor_status,
            const gss_name_t src_name,
            gss_name_t * dest_name
           )
{
    const gssnetlogon_name src = (const gssnetlogon_name)src_name;
    gssnetlogon_name dst = NULL;

    dst = calloc(1, sizeof(*dst));
    if (dst == NULL)
        goto fail;

    dst->NetbiosName.value = malloc(src->NetbiosName.length + 1);
    if (dst->NetbiosName.value == NULL)
        goto fail;
    memcpy(dst->NetbiosName.value, src->NetbiosName.value,
           src->NetbiosName.length);
    dst->NetbiosName.length = src->NetbiosName.length;
    ((char *)dst->NetbiosName.value)[dst->NetbiosName.length] = '\0';

    if (src->DnsName.length != 0) {
        dst->DnsName.value = malloc(src->DnsName.length + 1);
        if (dst->DnsName.value == NULL)
            goto fail;
        memcpy(dst->DnsName.value, src->DnsName.value, src->DnsName.length);
        dst->DnsName.length = src->DnsName.length;
        ((char *)dst->DnsName.value)[dst->DnsName.length] = '\0';
    }

    *minor_status = 0;
    *dest_name = (gss_name_t)dst;
    return GSS_S_COMPLETE;

fail:
    _netlogon_release_name(minor_status, (gss_name_t *)&dst);
    *minor_status = ENOMEM;
    return GSS_S_FAILURE;
}

