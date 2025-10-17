/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 25, 2024.
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
#include "gssdigest.h"

#ifdef ENABLE_SCRAM

static gssapi_mech_interface_desc scram_mech = {
    GMI_VERSION,
    "SCRAM-SHA1",
    {6, (void *)"\x2b\x06\x01\x05\x05\x0e"},
    0,
    _gss_scram_acquire_cred,
    _gss_scram_release_cred,
    _gss_scram_init_sec_context,
    _gss_scram_accept_sec_context,
    _gss_scram_process_context_token,
    _gss_scram_delete_sec_context,
    _gss_scram_context_time,
    NULL, /* get_mic */
    NULL, /* verify_mic */
    NULL, /* wrap */
    NULL, /* unwrap */
    _gss_scram_display_status,
    NULL,
    _gss_scram_compare_name,
    _gss_scram_display_name,
    _gss_scram_import_name,
    _gss_scram_export_name,
    _gss_scram_release_name,
    _gss_scram_inquire_cred,
    _gss_scram_inquire_context,
    NULL, /* wrap_size_limit */
    _gss_scram_add_cred,
    _gss_scram_inquire_cred_by_mech,
    _gss_scram_export_sec_context,
    _gss_scram_import_sec_context,
    _gss_scram_inquire_names_for_mech,
    _gss_scram_inquire_mechs_for_name,
    _gss_scram_canonicalize_name,
    _gss_scram_duplicate_name,
    _gss_scram_inquire_sec_context_by_oid,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL, /* wrap_iov */
    NULL, /* unwrap_iov */
    NULL, /* wrap_iov_length */
    NULL,
    NULL,
    NULL,
    _gss_scram_acquire_cred_ext,
    _gss_scram_iter_creds_f,
    _gss_scram_destroy_cred,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    0
};

#endif

gssapi_mech_interface
__gss_scram_initialize(void)
{
#ifdef ENABLE_SCRAM
    return &scram_mech;
#else
    return NULL;
#endif
}
