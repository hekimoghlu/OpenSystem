/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 13, 2022.
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

static gssapi_mech_interface_desc netlogon_mech = {
    GMI_VERSION,
    "netlogon",
    {6, rk_UNCONST("\x2a\x85\x70\x2b\x0e\x02") },
    0,
    _netlogon_acquire_cred,
    _netlogon_release_cred,
    _netlogon_init_sec_context,
    _netlogon_accept_sec_context,
    _netlogon_process_context_token,
    _netlogon_delete_sec_context,
    _netlogon_context_time,
    _netlogon_get_mic,
    _netlogon_verify_mic,
    NULL,
    NULL,
    _netlogon_display_status,
    NULL,
    _netlogon_compare_name,
    _netlogon_display_name,
    _netlogon_import_name,
    _netlogon_export_name,
    _netlogon_release_name,
    _netlogon_inquire_cred,
    _netlogon_inquire_context,
    _netlogon_wrap_size_limit,
    _netlogon_add_cred,
    _netlogon_inquire_cred_by_mech,
    _netlogon_export_sec_context,
    _netlogon_import_sec_context,
    _netlogon_inquire_names_for_mech,
    _netlogon_inquire_mechs_for_name,
    _netlogon_canonicalize_name,
    _netlogon_duplicate_name,
    NULL,
    NULL,
    NULL,
    _netlogon_set_cred_option,
    NULL,
    _netlogon_wrap_iov,
    _netlogon_unwrap_iov,
    _netlogon_wrap_iov_length,
    NULL,
    NULL,
    NULL,
    NULL,
    _netlogon_iter_creds_f,
    NULL,
    NULL
};

gssapi_mech_interface
__gss_netlogon_initialize(void)
{
    return &netlogon_mech;
}
