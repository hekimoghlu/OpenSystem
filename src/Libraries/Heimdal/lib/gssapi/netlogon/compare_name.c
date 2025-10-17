/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 6, 2022.
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

OM_uint32 _netlogon_compare_name
           (OM_uint32 * minor_status,
            const gss_name_t name1,
            const gss_name_t name2,
            int * name_equal
           )
{
    const gssnetlogon_name n1 = (const gssnetlogon_name)name1;
    const gssnetlogon_name n2 = (const gssnetlogon_name)name2;

    *name_equal = 0;

    if (n1->NetbiosName.value != NULL && n2->NetbiosName.value != NULL)
        *name_equal = (strcasecmp((char *)n1->NetbiosName.value,
                                  (char *)n2->NetbiosName.value) == 0);

    if (n1->DnsName.value != NULL && n2->DnsName.value != NULL)
        *name_equal = (strcasecmp((char *)n1->DnsName.value,
                                  (char *)n2->DnsName.value) == 0);

    *minor_status = 0;
    return GSS_S_COMPLETE;
}

