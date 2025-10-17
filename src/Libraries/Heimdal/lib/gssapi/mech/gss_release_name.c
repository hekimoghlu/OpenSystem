/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 29, 2023.
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
 * Free a name
 *
 * import_name can point to NULL or be NULL, or a pointer to a
 * gss_name_t structure. If it was a pointer to gss_name_t, the
 * pointer will be set to NULL on success and failure.
 *
 * @param minor_status minor status code
 * @param input_name name to free
 *
 * @returns a gss_error code, see gss_display_status() about printing
 *        the error code.
 *
 * @ingroup gssapi
 */
GSSAPI_LIB_FUNCTION OM_uint32 GSSAPI_LIB_CALL
gss_release_name(OM_uint32 * __nonnull minor_status,
		 __nullable gss_name_t * __nonnull input_name)
{
	struct _gss_name *name;

	*minor_status = 0;

	if (input_name == NULL || *input_name == NULL)
	    return GSS_S_COMPLETE;

	name = (struct _gss_name *) *input_name;
	*input_name = GSS_C_NO_NAME;

	_gss_mg_release_name(name);
	return (GSS_S_COMPLETE);
}
