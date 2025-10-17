/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 10, 2022.
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

/**
 * Create a representstion of a name suitable for display
 *
 * A name that is useful to print to user, not suitable for
 * authorization. For authorization use gss_authorize_localname(), or
 * gss_userok().
 *
 * @param minor_status minor status code returned
 * @param input_name name to be converted into a name
 * @param output_name_buffer output buffer with name, must be released with gss_release_buffer() on success.
 * @param output_name_type type OID of then name
 *
 * @returns GSS major status code
 *
 * @ingroup gssapi
 */

GSSAPI_LIB_FUNCTION OM_uint32 GSSAPI_LIB_CALL
gss_display_name(OM_uint32 *__nonnull minor_status,
    __nonnull const gss_name_t input_name,
    __nonnull gss_buffer_t output_name_buffer,
    __nullable gss_OID * __nullable output_name_type)
{
	OM_uint32 major_status;
	struct _gss_name *name = (struct _gss_name *) input_name;
	struct _gss_mechanism_name *mn;

	_mg_buffer_zero(output_name_buffer);
	if (output_name_type)
	    *output_name_type = GSS_C_NO_OID;

	if (name == NULL) {
		*minor_status = 0;
		return (GSS_S_BAD_NAME);
	}

	/*
	 * If we know it, copy the buffer used to import the name in
	 * the first place. Otherwise, ask all the MNs in turn if
	 * they can display the thing.
	 */
	if (name->gn_value.value) {
		output_name_buffer->value = malloc(name->gn_value.length);
		if (!output_name_buffer->value) {
			*minor_status = ENOMEM;
			return (GSS_S_FAILURE);
		}
		output_name_buffer->length = name->gn_value.length;
		memcpy(output_name_buffer->value, name->gn_value.value,
		    output_name_buffer->length);
		if (output_name_type)
			*output_name_type = &name->gn_type;

		*minor_status = 0;
		return (GSS_S_COMPLETE);
	} else {
		HEIM_SLIST_FOREACH(mn, &name->gn_mn, gmn_link) {
			major_status = mn->gmn_mech->gm_display_name(
				minor_status, mn->gmn_name,
				output_name_buffer,
				output_name_type);
			if (major_status == GSS_S_COMPLETE)
				return (GSS_S_COMPLETE);
		}
	}

	*minor_status = 0;
	return (GSS_S_FAILURE);
}
