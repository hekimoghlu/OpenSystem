/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 29, 2022.
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
#include "gssapi.h"
#include "nt_gss_common.h"

RCSID("$Id$");

/*
 * These are functions that are needed to interoperate with the
 * `Sample SSPI Code' in Windows 2000 RC1 SDK.
 */

/*
 * Write the `gss_buffer_t' in `buf' onto the fd `sock', but remember that
 * the length is written in little-endian-order.
 */

void
nt_write_token (int sock, gss_buffer_t buf)
{
    unsigned char net_len[4];
    uint32_t len;
    OM_uint32 min_stat;

    len = buf->length;

    net_len[0] = (len >>  0) & 0xFF;
    net_len[1] = (len >>  8) & 0xFF;
    net_len[2] = (len >> 16) & 0xFF;
    net_len[3] = (len >> 24) & 0xFF;

    if (write (sock, net_len, 4) != 4)
	err (1, "write");
    if (write (sock, buf->value, len) != len)
	err (1, "write");

    gss_release_buffer (&min_stat, buf);
}

/*
 *
 */

void
nt_read_token (int sock, gss_buffer_t buf)
{
    unsigned char net_len[4];
    uint32_t len;

    if (read(sock, net_len, 4) != 4)
	err (1, "read");
    len = (net_len[0] <<  0)
	| (net_len[1] <<  8)
	| (net_len[2] << 16)
	| (net_len[3] << 24);

    buf->length = len;
    buf->value  = malloc(len);
    if (read (sock, buf->value, len) != len)
	err (1, "read");
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
	fprintf (stderr, "%.*s\n",
		(int)status_string.length,
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
