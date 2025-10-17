/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 16, 2022.
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
#include "test_locl.h"
#include <gssapi.h>
#include <gssapi_krb5.h>
#include <gssapi_ntlm.h>
#include <gssapi_spnego.h>
#include "gss_common.h"

void
write_token (int sock, gss_buffer_t buf)
{
    uint32_t len, net_len;
    OM_uint32 min_stat;

    len = (uint32_t)buf->length;

    net_len = htonl(len);

    if (net_write (sock, &net_len, 4) != 4)
	err (1, "write");
    if (net_write (sock, buf->value, len) != len)
	err (1, "write");

    gss_release_buffer (&min_stat, buf);
}

static void
enet_read(int fd, void *buf, size_t len)
{
    ssize_t ret;

    ret = net_read (fd, buf, len);
    if (ret == 0)
	errx (1, "EOF in read");
    else if (ret < 0)
	errx (1, "read");
}

void
read_token (int sock, gss_buffer_t buf)
{
    uint32_t len, net_len;

    enet_read (sock, &net_len, 4);
    len = ntohl(net_len);
    buf->length = len;
    buf->value  = emalloc(len);
    enet_read (sock, buf->value, len);
}

void
gss_print_errors (int min_stat)
{
    OM_uint32 new_stat;
    OM_uint32 msg_ctx = 0;
    gss_buffer_desc status_string;
    OM_uint32 ret;

    do {
	ret = gss_display_status (&new_stat,
				  min_stat,
				  GSS_C_MECH_CODE,
				  GSS_C_NO_OID,
				  &msg_ctx,
				  &status_string);
	fprintf (stderr, "%.*s\n", (int)status_string.length,
		 (char *)status_string.value);
	gss_release_buffer (&new_stat, &status_string);
    } while (!GSS_ERROR(ret) && msg_ctx != 0);
}

void
gss_verr(int exitval, int status, const char *fmt, va_list ap)
{
    vwarnx (fmt, ap);
    gss_print_errors (status);
    exit (exitval);
}

void
gss_err(int exitval, int status, const char *fmt, ...)
{
    va_list args;

    va_start(args, fmt);
    gss_verr (exitval, status, fmt, args);
    va_end(args);
}

gss_OID
select_mech(const char *str)
{
    if (strcasecmp(str, "krb5") == 0)
	return GSS_KRB5_MECHANISM;
    else if (strcasecmp(str, "spnego") == 0)
	return GSS_SPNEGO_MECHANISM;
    else if (strcasecmp(str, "ntlm") == 0)
	return GSS_NTLM_MECHANISM;
    else if (strcasecmp(str, "no-oid") == 0)
	return GSS_C_NO_OID;
    else
	errx (1, "Unknown mechanism '%s' (spnego, krb5, no-oid)", str);
}

void
print_gss_name(const char *prefix, gss_name_t name)
{
    OM_uint32 maj_stat, min_stat;
    gss_buffer_desc name_token;

    maj_stat = gss_display_name (&min_stat,
				 name,
				 &name_token,
				 NULL);
    if (GSS_ERROR(maj_stat))
	gss_err (1, min_stat, "gss_display_name");

    fprintf (stderr, "%s `%.*s'\n", prefix,
	     (int)name_token.length,
	     (char *)name_token.value);

    gss_release_buffer (&min_stat, &name_token);

}
