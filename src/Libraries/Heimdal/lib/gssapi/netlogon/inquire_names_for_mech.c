/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 21, 2023.
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

OM_uint32 _netlogon_inquire_names_for_mech (
            OM_uint32 * minor_status,
            gss_const_OID mechanism,
            gss_OID_set * name_types
           )
{
    OM_uint32 ret, tmp;

    ret = gss_create_empty_oid_set(minor_status, name_types);
    if (ret != GSS_S_COMPLETE)
        return ret;

    ret = gss_add_oid_set_member(minor_status, GSS_NETLOGON_NT_NETBIOS_DNS_NAME, name_types);
    if (ret != GSS_S_COMPLETE) {
        gss_release_oid_set(&tmp, name_types);
        return ret;
    }

    *minor_status = 0;
    return GSS_S_COMPLETE;
}
