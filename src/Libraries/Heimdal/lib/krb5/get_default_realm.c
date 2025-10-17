/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 20, 2025.
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
#include "krb5_locl.h"

krb5_error_code
_krb5_array_to_realms(krb5_context context, heim_array_t array, krb5_realm **realms)
{
    size_t n, len;

    len = heim_array_get_length(array);
    
    *realms = calloc(len + 1, sizeof((*realms)[0]));
    for (n = 0; n < len; n++) {
	heim_string_t s = heim_array_copy_value(array, n);
	if (s) {
	    (*realms)[n] = heim_string_copy_utf8(s);
	    heim_release(s);
	}
	if ((*realms)[n] == NULL) {
	    krb5_free_host_realm(context, *realms);
	    krb5_set_error_message(context, ENOMEM,
				   N_("malloc: out of memory", ""));
	    *realms = NULL;
	    return ENOMEM;
	}
    }
    (*realms)[n] = NULL;

    return 0;
}

/*
 * Get the list of default realms and make sure there is at least
 * one realm configured.
 */

static krb5_error_code
get_default_realms(krb5_context context)
{
    if (context->default_realms == NULL ||
	heim_array_get_length(context->default_realms) == 0)
    {
	krb5_error_code ret = krb5_set_default_realm(context, NULL);
	if (ret)
	    return KRB5_CONFIG_NODEFREALM;
    }
    
    if (context->default_realms == NULL ||
	heim_array_get_length(context->default_realms) == 0)
    {
	krb5_set_error_message(context, KRB5_CONFIG_NODEFREALM,
			       N_("No default realm found", ""));
	return KRB5_CONFIG_NODEFREALM;
    }

    return 0;
}

/*
 * Return a NULL-terminated list of default realms in `realms'.
 * Free this memory with krb5_free_host_realm.
 */

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_get_default_realms (krb5_context context,
			 krb5_realm **realms)
{
    krb5_error_code ret;

    ret = get_default_realms(context);
    if (ret)
	return ret;
    
    return _krb5_array_to_realms(context, context->default_realms, realms);
}

/*
 * Return the first default realm.  For compatibility.
 */

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_get_default_realm(krb5_context context,
		       krb5_realm *realm)
{
    krb5_error_code ret;
    heim_string_t s;
    
    ret = get_default_realms(context);
    if (ret)
	return ret;
    
    s = heim_array_copy_value(context->default_realms, 0);
    if (s) {
	*realm = heim_string_copy_utf8(s);
	heim_release(s);
    }
    if (s == NULL || *realm == NULL) {
	krb5_set_error_message(context, ENOMEM,
			       N_("malloc: out of memory", ""));
	return ENOMEM;
    }
    
    return 0;
}
