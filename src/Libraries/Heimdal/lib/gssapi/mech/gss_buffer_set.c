/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 12, 2023.
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
gss_create_empty_buffer_set
	   (OM_uint32 * __nonnull minor_status,
	    __nullable gss_buffer_set_t *__nonnull buffer_set)
{
    gss_buffer_set_t set;

    set = (gss_buffer_set_desc *) malloc(sizeof(*set));
    if (set == GSS_C_NO_BUFFER_SET) {
	*minor_status = ENOMEM;
	return GSS_S_FAILURE;
    }

    set->count = 0;
    set->elements = NULL;

    *buffer_set = set;

    *minor_status = 0;
    return GSS_S_COMPLETE;
}

GSSAPI_LIB_FUNCTION OM_uint32 GSSAPI_LIB_CALL
gss_add_buffer_set_member
	   (OM_uint32 * __nonnull minor_status,
	    __nonnull const gss_buffer_t member_buffer,
	    __nonnull gss_buffer_set_t *__nonnull buffer_set)
{
    gss_buffer_set_t set;
    gss_buffer_t p;
    OM_uint32 ret;

    if (*buffer_set == GSS_C_NO_BUFFER_SET) {
	ret = gss_create_empty_buffer_set(minor_status,
					  buffer_set);
	if (ret) {
	    return ret;
	}
    }

    set = *buffer_set;
    set->elements = realloc(set->elements,
			    (set->count + 1) * sizeof(set->elements[0]));
    if (set->elements == NULL) {
	*minor_status = ENOMEM;
	return GSS_S_FAILURE;
    }

    p = &set->elements[set->count];

    p->value = malloc(member_buffer->length);
    if (p->value == NULL) {
	*minor_status = ENOMEM;
	return GSS_S_FAILURE;
    }
    memcpy(p->value, member_buffer->value, member_buffer->length);
    p->length = member_buffer->length;

    set->count++;

    *minor_status = 0;
    return GSS_S_COMPLETE;
}

GSSAPI_LIB_FUNCTION OM_uint32 GSSAPI_LIB_CALL
gss_release_buffer_set(OM_uint32 * __nonnull minor_status,
		       __nullable gss_buffer_set_t * __nonnull buffer_set)
{
    size_t i;
    OM_uint32 minor;

    *minor_status = 0;

    if (*buffer_set == GSS_C_NO_BUFFER_SET)
	return GSS_S_COMPLETE;

    for (i = 0; i < (*buffer_set)->count; i++)
	gss_release_buffer(&minor, &((*buffer_set)->elements[i]));

    free((*buffer_set)->elements);

    (*buffer_set)->elements = NULL;
    (*buffer_set)->count = 0;

    free(*buffer_set);
    *buffer_set = GSS_C_NO_BUFFER_SET;

    return GSS_S_COMPLETE;
}

