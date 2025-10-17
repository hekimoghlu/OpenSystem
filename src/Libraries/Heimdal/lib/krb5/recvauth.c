/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 16, 2023.
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

/*
 * See `sendauth.c' for the format.
 */

static krb5_boolean
match_exact(void *data, const char *appl_version)
{
    return strcmp(data, appl_version) == 0;
}

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_recvauth(krb5_context context,
	      krb5_auth_context *auth_context,
	      krb5_pointer p_fd,
	      const char *appl_version,
	      krb5_principal server,
	      int32_t flags,
	      krb5_keytab keytab,
	      krb5_ticket **ticket)
{
    return krb5_recvauth_match_version(context, auth_context, p_fd,
				       match_exact, (void *)appl_version,
				       server, flags,
				       keytab, ticket);
}

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_recvauth_match_version(krb5_context context,
			    krb5_auth_context *auth_context,
			    krb5_pointer p_fd,
			    krb5_boolean (*match_appl_version)(void *, const char*),
			    void *match_data,
			    krb5_principal server,
			    int32_t flags,
			    krb5_keytab keytab,
			    krb5_ticket **ticket)
{
    krb5_error_code ret;
    const char *version = KRB5_SENDAUTH_VERSION;
    char her_version[sizeof(KRB5_SENDAUTH_VERSION)];
    char *her_appl_version;
    uint32_t len;
    u_char repl;
    krb5_data data;
    krb5_flags ap_options;
    ssize_t n;

    /*
     * If there are no addresses in auth_context, get them from `fd'.
     */

    if (*auth_context == NULL) {
	ret = krb5_auth_con_init (context, auth_context);
	if (ret)
	    return ret;
    }

    ret = krb5_auth_con_setaddrs_from_fd (context,
					  *auth_context,
					  p_fd);
    if (ret)
	return ret;

    if(!(flags & KRB5_RECVAUTH_IGNORE_VERSION)) {
	n = krb5_net_read (context, p_fd, &len, 4);
	if (n < 0) {
	    ret = errno;
	    krb5_set_error_message(context, ret, "read: %s", strerror(ret));
	    return ret;
	}
	if (n == 0) {
	    krb5_set_error_message(context, KRB5_SENDAUTH_BADAUTHVERS,
				   N_("Failed to receive sendauth data", ""));
	    return KRB5_SENDAUTH_BADAUTHVERS;
	}
	len = ntohl(len);
	if (len != sizeof(her_version)
	    || krb5_net_read (context, p_fd, her_version, len) != (ssize_t)len
	    || strncmp (version, her_version, len)) {
	    repl = 1;
	    krb5_net_write (context, p_fd, &repl, 1);
	    krb5_clear_error_message (context);
	    return KRB5_SENDAUTH_BADAUTHVERS;
	}
    }

    n = krb5_net_read (context, p_fd, &len, 4);
    if (n < 0) {
	ret = errno;
	krb5_set_error_message(context, ret, "read: %s", strerror(ret));
	return ret;
    }
    if (n == 0) {
	krb5_clear_error_message (context);
	return KRB5_SENDAUTH_BADAPPLVERS;
    }
    len = ntohl(len);
    if (len > UINT_MAX / 16) {
	krb5_set_error_message(context, ERANGE,
			       N_("packet to large", ""));
	return ERANGE;
    }
    her_appl_version = malloc (len);
    if (her_appl_version == NULL) {
	repl = 2;
	krb5_net_write (context, p_fd, &repl, 1);
	krb5_set_error_message(context, ENOMEM,
			       N_("malloc: out of memory", ""));
	return ENOMEM;
    }
    if (krb5_net_read (context, p_fd, her_appl_version, len) != (ssize_t)len
	|| !(*match_appl_version)(match_data, her_appl_version)) {
	repl = 2;
	krb5_net_write (context, p_fd, &repl, 1);
	krb5_set_error_message(context, KRB5_SENDAUTH_BADAPPLVERS,
			       N_("wrong sendauth version (%s)", ""),
			       her_appl_version);
	free (her_appl_version);
	return KRB5_SENDAUTH_BADAPPLVERS;
    }
    free (her_appl_version);

    repl = 0;
    if (krb5_net_write (context, p_fd, &repl, 1) != 1) {
	ret = errno;
	krb5_set_error_message(context, ret, "write: %s", strerror(ret));
	return ret;
    }

    krb5_data_zero (&data);
    ret = krb5_read_message (context, p_fd, &data);
    if (ret) {
	krb5_set_error_message(context, ret, "krb5_recvauth: client closed connection");
	return ret;
    }

    ret = krb5_rd_req (context,
		       auth_context,
		       &data,
		       server,
		       keytab,
		       &ap_options,
		       ticket);
    krb5_data_free (&data);
    if (ret) {
	krb5_data error_data;
	krb5_error_code ret2;

	ret2 = krb5_mk_error (context,
			      ret,
			      NULL,
			      NULL,
			      NULL,
			      server,
			      NULL,
			      NULL,
			      &error_data);
	if (ret2 == 0) {
	    krb5_write_message (context, p_fd, &error_data);
	    krb5_data_free (&error_data);
	}
	return ret;
    }

    len = 0;
    if (krb5_net_write (context, p_fd, &len, 4) != 4) {
	ret = errno;
	krb5_set_error_message(context, ret, "write: %s", strerror(ret));
	krb5_free_ticket(context, *ticket);
	*ticket = NULL;
	return ret;
    }

    if (ap_options & AP_OPTS_MUTUAL_REQUIRED) {
	ret = krb5_mk_rep (context, *auth_context, &data);
	if (ret) {
	    krb5_free_ticket(context, *ticket);
	    *ticket = NULL;
	    return ret;
	}

	ret = krb5_write_message (context, p_fd, &data);
	if (ret) {
	    krb5_set_error_message(context, ret, "krb5_recvauth: server closed connection");
	    krb5_free_ticket(context, *ticket);
	    *ticket = NULL;
	    return ret;
	}
	krb5_data_free (&data);
    }
    return 0;
}
