/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 29, 2023.
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
#include "gsskrb5_locl.h"

static OM_uint32
import_krb5_name(OM_uint32 *minor_status,
		 gss_const_OID mech,
		 const gss_buffer_t input_name_buffer,
		 gss_const_OID input_name_type,
		 gss_name_t *output_name)
{
    krb5_context context;
    krb5_principal princ;
    krb5_error_code ret;
    char *tmp;

    GSSAPI_KRB5_INIT (&context);

    tmp = malloc (input_name_buffer->length + 1);
    if (tmp == NULL) {
	*minor_status = ENOMEM;
	return GSS_S_FAILURE;
    }
    memcpy (tmp,
	    input_name_buffer->value,
	    input_name_buffer->length);
    tmp[input_name_buffer->length] = '\0';

    if (tmp[0] == '@') {
	princ = calloc(1, sizeof(*princ));
	if (princ == NULL) {
	    free(tmp);
	    *minor_status = ENOMEM;
	    return GSS_S_FAILURE;
	}

	princ->realm = strdup(&tmp[1]);
	if (princ->realm == NULL) {
	    free(tmp);
	    free(princ);
	    return GSS_S_FAILURE;
	}
    } else {
	ret = krb5_parse_name (context, tmp, &princ);
	if (ret) {
	    free(tmp);
	    *minor_status = ret;

	    if (ret == KRB5_PARSE_ILLCHAR || ret == KRB5_PARSE_MALFORMED)
		return GSS_S_BAD_NAME;

	    return GSS_S_FAILURE;
	}
    }

    if (mech && gss_oid_equal(mech, GSS_PKU2U_MECHANISM) && strchr(tmp, '@') == NULL)
	krb5_principal_set_realm(context, princ, KRB5_PKU2U_REALM_NAME);

    free(tmp);

    if (princ->name.name_string.len == 2 &&
	gss_oid_equal(input_name_type, GSS_KRB5_NT_PRINCIPAL_NAME_REFERRAL))
	krb5_principal_set_type(context, princ, KRB5_NT_GSS_HOSTBASED_SERVICE);

    *output_name = (gss_name_t)princ;
    return GSS_S_COMPLETE;
}

static OM_uint32
import_krb5_principal(OM_uint32 *minor_status,
		      gss_const_OID mech,
		      const gss_buffer_t input_name_buffer,
		      gss_const_OID input_name_type,
		      gss_name_t *output_name)
{
    krb5_context context;
    krb5_principal *princ, res = NULL;
    OM_uint32 ret;

    GSSAPI_KRB5_INIT (&context);

    princ = (krb5_principal *)input_name_buffer->value;

    ret = krb5_copy_principal(context, *princ, &res);
    if (ret) {
	*minor_status = ret;
	return GSS_S_FAILURE;
    }
    *output_name = (gss_name_t)res;
    return GSS_S_COMPLETE;
}


OM_uint32
_gsskrb5_canon_name(OM_uint32 *minor_status, krb5_context context,
		    int use_dns, krb5_const_principal sourcename, gss_name_t targetname,
		    krb5_principal *out)
{
    krb5_principal p = (krb5_principal)targetname;
    krb5_error_code ret;
    char *hostname = NULL, *service;

    *minor_status = 0;

    /* If its not a hostname */
    if (krb5_principal_get_type(context, p) != KRB5_NT_GSS_HOSTBASED_SERVICE) {
	ret = krb5_copy_principal(context, p, out);
    } else if (!use_dns) {
	ret = krb5_copy_principal(context, p, out);
	if (ret)
	    goto out;
	krb5_principal_set_type(context, *out, KRB5_NT_SRV_HST);
	if (sourcename)
	    ret = krb5_principal_set_realm(context, *out, sourcename->realm);
    } else {
	if (p->name.name_string.len == 0)
	    return GSS_S_BAD_NAME;
	else if (p->name.name_string.len > 1)
	    hostname = p->name.name_string.val[1];

	service = p->name.name_string.val[0];

	ret = krb5_sname_to_principal(context,
				      hostname,
				      service,
				      KRB5_NT_SRV_HST,
				      out);
    }

 out:
    if (ret) {
	*minor_status = ret;
	return GSS_S_FAILURE;
    }

    return 0;
}


static OM_uint32
import_hostbased_name (OM_uint32 *minor_status,
		       gss_const_OID mech,
		       const gss_buffer_t input_name_buffer,
		       gss_const_OID input_name_type,
		       gss_name_t *output_name)
{
    krb5_context context;
    krb5_principal princ = NULL;
    krb5_error_code kerr;
    char *tmp, *p, *host = NULL, *realm = NULL;

    if (gss_oid_equal(mech, GSS_PKU2U_MECHANISM))
	realm = KRB5_PKU2U_REALM_NAME;
    else
	realm = KRB5_GSS_REFERALS_REALM_NAME; /* should never hit the network */

    GSSAPI_KRB5_INIT (&context);

    tmp = malloc (input_name_buffer->length + 1);
    if (tmp == NULL) {
	*minor_status = ENOMEM;
	return GSS_S_FAILURE;
    }
    memcpy (tmp,
	    input_name_buffer->value,
	    input_name_buffer->length);
    tmp[input_name_buffer->length] = '\0';

    p = strchr (tmp, '@');
    if (p != NULL && p[1] != '\0') {
	size_t len;

	*p = '\0';
	host = p + 1;

	/*
	 * Squash any trailing . on the hostname since that is jolly
	 * good to have when looking up a DNS name (qualified), but
	 * its no good to have in the kerberos principal since those
	 * are supposed to be in qualified format already.
	 */

	len = strlen(host);
	if (len > 0 && host[len - 1] == '.')
	    host[len - 1] = '\0';
    } else {
	host = KRB5_GSS_HOSTBASED_SERVICE_NAME;
    }

    kerr = krb5_make_principal(context, &princ, realm, tmp, host, NULL);
    free (tmp);
    *minor_status = kerr;
    if (kerr == KRB5_PARSE_ILLCHAR || kerr == KRB5_PARSE_MALFORMED)
	return GSS_S_BAD_NAME;
    else if (kerr)
	return GSS_S_FAILURE;

    krb5_principal_set_type(context, princ, KRB5_NT_GSS_HOSTBASED_SERVICE);
    *output_name = (gss_name_t)princ;

    return 0;
}

static OM_uint32
import_dn_name(OM_uint32 *minor_status,
	       gss_const_OID mech,
	       const gss_buffer_t input_name_buffer,
	       gss_const_OID input_name_type,
	       gss_name_t *output_name)
{
    /* XXX implement me */
    *output_name = NULL;
    *minor_status = 0;
    return GSS_S_FAILURE;
}

static OM_uint32
import_pku2u_export_name(OM_uint32 *minor_status,
			 gss_const_OID mech,
			 const gss_buffer_t input_name_buffer,
			 gss_const_OID input_name_type,
			 gss_name_t *output_name)
{
    /* XXX implement me */
    *output_name = NULL;
    *minor_status = 0;
    return GSS_S_FAILURE;
}

static OM_uint32
import_uuid_name(OM_uint32 *minor_status,
		 gss_const_OID mech,
		 const gss_buffer_t input_name_buffer,
		 gss_const_OID input_name_type,
		 gss_name_t *output_name)
{
    krb5_context context;
    krb5_error_code ret;
    krb5_principal princ;
    char uuid[36 + 1];

    GSSAPI_KRB5_INIT(&context);
    
    if (input_name_buffer->length < sizeof(uuid) - 1) {
	*minor_status = 0;
	return GSS_S_BAD_NAME;
    }
    
    memcpy(uuid, input_name_buffer->value, sizeof(uuid) - 1);
    uuid[sizeof(uuid) - 1] = '\0';
    
    /* validate that uuid is only uuid chars and the right length*/
    if (strspn(uuid, "0123456789abcdefABCDEF-") != 36) {
	*minor_status = 0;
	return GSS_S_BAD_NAME;
    }
    
    ret = krb5_make_principal(context, &princ, "UUID", uuid, NULL);
    if (ret) {
	*minor_status = ret;
	return GSS_S_FAILURE;
    }
    krb5_principal_set_type(context, princ, KRB5_NT_CACHE_UUID);
    
    *output_name = (gss_name_t)princ;
    *minor_status = 0;

    return GSS_S_COMPLETE;
}

static struct _gss_name_type krb5_names[] = {
    { GSS_C_NT_HOSTBASED_SERVICE, import_hostbased_name },
    { GSS_C_NT_HOSTBASED_SERVICE_X, import_hostbased_name },
    { GSS_KRB5_NT_PRINCIPAL, import_krb5_principal},
    { GSS_C_NO_OID, import_krb5_name },
    { GSS_C_NT_USER_NAME, import_krb5_name },
    { GSS_KRB5_NT_PRINCIPAL_NAME, import_krb5_name },
    { GSS_KRB5_NT_PRINCIPAL_NAME_REFERRAL, import_krb5_name },
    { GSS_C_NT_EXPORT_NAME, import_krb5_name },
    { GSS_C_NT_UUID, import_uuid_name },
    { NULL, NULL }
};

static struct _gss_name_type pku2u_names[] = {
    { GSS_C_NT_HOSTBASED_SERVICE, import_hostbased_name },
    { GSS_C_NT_HOSTBASED_SERVICE_X, import_hostbased_name },
    { GSS_C_NO_OID, import_krb5_name },
    { GSS_C_NT_USER_NAME, import_krb5_name },
    { GSS_KRB5_NT_PRINCIPAL_NAME, import_krb5_name },
    { GSS_C_NT_DN, import_dn_name },
    { GSS_C_NT_EXPORT_NAME, import_pku2u_export_name },
    { GSS_C_NT_UUID, import_uuid_name },
    { NULL, NULL }
};

static struct _gss_name_type iakerb_names[] = {
    { GSS_C_NT_HOSTBASED_SERVICE, import_hostbased_name },
    { GSS_C_NT_HOSTBASED_SERVICE_X, import_hostbased_name },
    { GSS_C_NO_OID, import_krb5_name },
    { GSS_C_NT_USER_NAME, import_krb5_name },
    { GSS_KRB5_NT_PRINCIPAL_NAME, import_krb5_name },
    { GSS_KRB5_NT_PRINCIPAL_NAME_REFERRAL, import_krb5_name },
    { GSS_C_NT_EXPORT_NAME, import_krb5_name },
    { GSS_C_NT_UUID, import_uuid_name },
    { NULL, NULL }
};

OM_uint32 GSSAPI_CALLCONV _gsskrb5_import_name
           (OM_uint32 * minor_status,
            const gss_buffer_t input_name_buffer,
            gss_const_OID input_name_type,
            gss_name_t * output_name
           )
{
    return _gss_mech_import_name(minor_status, GSS_KRB5_MECHANISM,
				 krb5_names, input_name_buffer,
				 input_name_type, output_name);
}

OM_uint32 _gsspku2u_import_name
           (OM_uint32 * minor_status,
            const gss_buffer_t input_name_buffer,
            gss_const_OID input_name_type,
            gss_name_t * output_name
           )
{
    return _gss_mech_import_name(minor_status, GSS_PKU2U_MECHANISM,
				 pku2u_names, input_name_buffer,
				 input_name_type, output_name);
}

OM_uint32
_gssiakerb_import_name(OM_uint32 * minor_status,
		       const gss_buffer_t input_name_buffer,
		       gss_const_OID input_name_type,
		       gss_name_t * output_name)
{
    return _gss_mech_import_name(minor_status, GSS_IAKERB_MECHANISM,
				 iakerb_names, input_name_buffer,
				 input_name_type, output_name);
}

OM_uint32
_gsskrb5_inquire_names_for_mech (OM_uint32 * minor_status,
				 gss_const_OID mechanism,
				 gss_OID_set * name_types)
{
    return _gss_mech_inquire_names_for_mech(minor_status, krb5_names,
					    name_types);
}

OM_uint32
_gsspku2u_inquire_names_for_mech (OM_uint32 * minor_status,
				  gss_const_OID mechanism,
				  gss_OID_set * name_types)
{
    return _gss_mech_inquire_names_for_mech(minor_status, pku2u_names,
					    name_types);
}

OM_uint32
_gssiakerb_inquire_names_for_mech (OM_uint32 * minor_status,
				   gss_const_OID mechanism,
				   gss_OID_set * name_types)
{
    return _gss_mech_inquire_names_for_mech(minor_status, iakerb_names,
					    name_types);
}
