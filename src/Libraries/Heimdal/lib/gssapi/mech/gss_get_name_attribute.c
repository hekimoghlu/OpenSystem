/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 8, 2023.
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
#include "mech_locl.h"

GSSAPI_LIB_FUNCTION OM_uint32 GSSAPI_LIB_CALL
gss_get_name_attribute(OM_uint32 *__nonnull minor_status,
		       __nonnull gss_name_t input_name,
		       __nonnull gss_buffer_t attr,
		       int *__nullable authenticated,
		       int *__nullable complete,
		       __nullable gss_buffer_t value,
		       __nullable gss_buffer_t display_value,
		       int *__nonnull more)
{
    OM_uint32 major_status = GSS_S_UNAVAILABLE;
    struct _gss_name *name = (struct _gss_name *) input_name;
    struct _gss_mechanism_name *mn;

    *minor_status = 0;
    if (authenticated != NULL)
        *authenticated = 0;
    if (complete != NULL)
        *complete = 0;
    _mg_buffer_zero(value);
    _mg_buffer_zero(display_value);


    if (input_name == GSS_C_NO_NAME)
        return GSS_S_BAD_NAME;

    _gss_mg_check_name(input_name);


    HEIM_SLIST_FOREACH(mn, &name->gn_mn, gmn_link) {
        gssapi_mech_interface m = mn->gmn_mech;

        if (!m->gm_get_name_attribute)
            continue;

        major_status = m->gm_get_name_attribute(minor_status,
                                                mn->gmn_name,
                                                attr,
                                                authenticated,
                                                complete,
                                                value,
                                                display_value,
                                                more);
        if (GSS_ERROR(major_status))
            _gss_mg_error(m, *minor_status);
        else
            break;
    }

    return major_status;
}
