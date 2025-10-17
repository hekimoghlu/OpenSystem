/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 17, 2021.
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
#include <ctype.h>

OM_uint32 _netlogon_import_name
           (OM_uint32 * minor_status,
            const gss_buffer_t input_name_buffer,
            gss_const_OID input_name_type,
            gss_name_t * output_name
           )
{
    gssnetlogon_name name;
    const char *netbiosName;
    const char *dnsName = NULL;
    size_t len, i;

    if (!gss_oid_equal(input_name_type, GSS_NETLOGON_NT_NETBIOS_DNS_NAME)) {
        return GSS_S_BAD_NAME;
    }

    /* encoding is NetBIOS name \0 DNS name \0 */

    netbiosName = input_name_buffer->value;
    len = strlen(netbiosName);
    if (len < input_name_buffer->length)
        dnsName = netbiosName + len + 1;

    name = (gssnetlogon_name)calloc(1, sizeof(*name));
    if (name == NULL)
        goto cleanup;

    name->NetbiosName.value = malloc(len + 1);
    if (name->NetbiosName.value == NULL)
        goto cleanup;
    memcpy(name->NetbiosName.value, netbiosName, len + 1);
    name->NetbiosName.length = len;

    /* normalise name to uppercase XXX UTF-8 OK? */
    for (i = 0; i < len; i++) {
        ((char *)name->NetbiosName.value)[i] =
            toupper(((char *)name->NetbiosName.value)[i]);
    }

    if (dnsName != NULL && dnsName[0] != '\0') {
        name->DnsName.value = strdup(dnsName);
        if (name->DnsName.value == NULL)
            goto cleanup;
        name->DnsName.length = strlen(dnsName);
    }

    *output_name = (gss_name_t)name;
    *minor_status = 0;
    return GSS_S_COMPLETE;

cleanup:
    _netlogon_release_name(minor_status, (gss_name_t *)&name);
    *minor_status = ENOMEM;
    return GSS_S_FAILURE;
}

