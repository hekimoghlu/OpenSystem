/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 1, 2024.
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
#include "ntlm.h"

gss_name_t
_gss_ntlm_create_name(OM_uint32 *minor_status,
		      const char *user, const char *domain, int flags)
{
    ntlm_name n;
    n = calloc(1, sizeof(*n));
    if (n == NULL) {
	*minor_status = ENOMEM;
	return NULL;
    }

    n->user = strdup(user);
    n->domain = strdup(domain);
    n->flags = flags;

    if (n->user == NULL || n->domain == NULL) {
	free(n->user);
	free(n->domain);
	free(n);
	*minor_status = ENOMEM;
	return NULL;
    }

    return (gss_name_t)n;
}

static OM_uint32
anon_name(OM_uint32 *minor_status,
	  gss_const_OID mech,
	  const gss_buffer_t input_name_buffer,
	  gss_const_OID input_name_type,
	  gss_name_t *output_name)
{
    *output_name = _gss_ntlm_create_name(minor_status, "", "", NTLM_ANON_NAME);
    if (*output_name == NULL)
	return GSS_S_FAILURE;
    return GSS_S_COMPLETE;
}

static OM_uint32
hostbased_name(OM_uint32 *minor_status,
	       gss_const_OID mech,
	       const gss_buffer_t input_name_buffer,
	       gss_const_OID input_name_type,
	       gss_name_t *output_name)
{
    char *name, *p;

    name = malloc(input_name_buffer->length + 1);
    if (name == NULL) {
	*minor_status = ENOMEM;
	return GSS_S_FAILURE;
    }
    memcpy(name, input_name_buffer->value, input_name_buffer->length);
    name[input_name_buffer->length] = '\0';

    /* find "domain" part of the name and uppercase it */
    p = strchr(name, '@');
    if (p) {
	p[0] = '\0';
	p++;
    } else {
	p = "";
    }

    *output_name = _gss_ntlm_create_name(minor_status, name, p, 0);
    free(name);
    if (*output_name == NULL)
	return GSS_S_FAILURE;

    return GSS_S_COMPLETE;
}

static OM_uint32
parse_name(OM_uint32 *minor_status,
	   gss_const_OID mech,
	   int domain_required,
	   const gss_buffer_t input_name_buffer,
	   gss_const_OID input_name_type,
	   gss_name_t *output_name)
{
    char *name, *p, *user, *domain;

    if (memchr(input_name_buffer->value, '@', input_name_buffer->length) != NULL)
	return hostbased_name(minor_status, mech, input_name_buffer,
			      input_name_type, output_name);

    name = malloc(input_name_buffer->length + 1);
    if (name == NULL) {
	*minor_status = ENOMEM;
	return GSS_S_FAILURE;
    }
    memcpy(name, input_name_buffer->value, input_name_buffer->length);
    name[input_name_buffer->length] = '\0';

    /* find "domain" part of the name and uppercase it */
    p = strchr(name, '\\');
    if (p) {
	p[0] = '\0';
	user = p + 1;
	domain = name;
	strupr(domain);
    } else if (!domain_required) {
	user = name;
	domain = ""; /* no domain */
    } else {
	free(name);
	*minor_status = HNTLM_ERR_MISSING_NAME_SEPARATOR;
	return gss_mg_set_error_string(GSS_NTLM_MECHANISM, GSS_S_BAD_NAME,
				       HNTLM_ERR_MISSING_NAME_SEPARATOR,
				       "domain requested but missing name");
    }

    *output_name = _gss_ntlm_create_name(minor_status, user, domain, 0);
    free(name);
    if (*output_name == NULL)
	return GSS_S_FAILURE;

    return GSS_S_COMPLETE;
}

static OM_uint32
user_name(OM_uint32 *minor_status,
	  gss_const_OID mech,
	  const gss_buffer_t input_name_buffer,
	  gss_const_OID input_name_type,
	  gss_name_t *output_name)
{
    return parse_name(minor_status, mech, 0, input_name_buffer, input_name_type, output_name);
}

static OM_uint32
parse_ntlm_name(OM_uint32 *minor_status,
		gss_const_OID mech,
		const gss_buffer_t input_name_buffer,
		gss_const_OID input_name_type,
		gss_name_t *output_name)
{
    return parse_name(minor_status, mech, 1, input_name_buffer, input_name_type, output_name);
}

static OM_uint32
export_name(OM_uint32 *minor_status,
	    gss_const_OID mech,
	    const gss_buffer_t input_name_buffer,
	    gss_const_OID input_name_type,
	    gss_name_t *output_name)
{
    return parse_name(minor_status, mech, 1, input_name_buffer, input_name_type, output_name);
}

static struct _gss_name_type ntlm_names[] = {
    { GSS_C_NT_ANONYMOUS, anon_name},
    { GSS_C_NT_HOSTBASED_SERVICE, hostbased_name},
    { GSS_C_NT_USER_NAME, user_name },
    { GSS_C_NT_NTLM, parse_ntlm_name },
    { GSS_C_NT_EXPORT_NAME, export_name },
    { NULL }
};


OM_uint32 _gss_ntlm_import_name
           (OM_uint32 * minor_status,
            const gss_buffer_t input_name_buffer,
            gss_const_OID input_name_type,
            gss_name_t * output_name
           )
{
    return _gss_mech_import_name(minor_status, GSS_NTLM_MECHANISM,
				 ntlm_names, input_name_buffer,
				 input_name_type, output_name);
}

OM_uint32 _gss_ntlm_inquire_names_for_mech (
            OM_uint32 * minor_status,
            gss_const_OID mechanism,
            gss_OID_set * name_types
           )
{
    return _gss_mech_inquire_names_for_mech(minor_status, ntlm_names,
					    name_types);
}
