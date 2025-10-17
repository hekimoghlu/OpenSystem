/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 28, 2025.
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
RCSID("$Id$");

krb5_context context;

static int
proto (int sock, const char *service)
{
    struct sockaddr_in remote, local;
    socklen_t addrlen;
    krb5_address remote_addr, local_addr;
    krb5_ccache ccache;
    krb5_auth_context auth_context;
    krb5_error_code status;
    krb5_data packet;
    krb5_data data;
    krb5_data client_name;
    krb5_creds in_creds, *out_creds;

    addrlen = sizeof(local);
    if (getsockname (sock, (struct sockaddr *)&local, &addrlen) < 0
	|| addrlen != sizeof(local))
	err (1, "getsockname)");

    addrlen = sizeof(remote);
    if (getpeername (sock, (struct sockaddr *)&remote, &addrlen) < 0
	|| addrlen != sizeof(remote))
	err (1, "getpeername");

    status = krb5_auth_con_init (context, &auth_context);
    if (status)
	krb5_err(context, 1, status, "krb5_auth_con_init");

    local_addr.addr_type = AF_INET;
    local_addr.address.length = sizeof(local.sin_addr);
    local_addr.address.data   = &local.sin_addr;

    remote_addr.addr_type = AF_INET;
    remote_addr.address.length = sizeof(remote.sin_addr);
    remote_addr.address.data   = &remote.sin_addr;

    status = krb5_auth_con_setaddrs (context,
				     auth_context,
				     &local_addr,
				     &remote_addr);
    if (status)
	krb5_err(context, 1, status, "krb5_auth_con_setaddr");

    status = krb5_read_message(context, &sock, &client_name);
    if(status)
	krb5_err(context, 1, status, "krb5_read_message");

    memset(&in_creds, 0, sizeof(in_creds));
    status = krb5_cc_default(context, &ccache);
    if(status)
	krb5_err(context, 1, status, "krb5_cc_default");
    status = krb5_cc_get_principal(context, ccache, &in_creds.client);
    if(status)
	krb5_err(context, 1, status, "krb5_cc_get_principal");

    status = krb5_read_message(context, &sock, &in_creds.second_ticket);
    if(status)
	krb5_err(context, 1, status, "krb5_read_message");

    status = krb5_parse_name(context, client_name.data, &in_creds.server);
    if(status)
	krb5_err(context, 1, status, "krb5_parse_name");

    status = krb5_get_credentials(context, KRB5_GC_USER_USER, ccache,
				  &in_creds, &out_creds);
    if(status)
	krb5_err(context, 1, status, "krb5_get_credentials");

    status = krb5_cc_default(context, &ccache);
    if(status)
	krb5_err(context, 1, status, "krb5_cc_default");

    status = krb5_sendauth(context,
			   &auth_context,
			   &sock,
			   VERSION,
			   in_creds.client,
			   in_creds.server,
			   AP_OPTS_USE_SESSION_KEY,
			   NULL,
			   out_creds,
			   ccache,
			   NULL,
			   NULL,
			   NULL);

    if (status)
	krb5_err(context, 1, status, "krb5_sendauth");

    {
	char *str;
	krb5_unparse_name(context, in_creds.server, &str);
	printf ("User is `%s'\n", str);
	free(str);
	krb5_unparse_name(context, in_creds.client, &str);
	printf ("Server is `%s'\n", str);
	free(str);
    }

    krb5_data_zero (&data);
    krb5_data_zero (&packet);

    status = krb5_read_message(context, &sock, &packet);
    if(status)
	krb5_err(context, 1, status, "krb5_read_message");

    status = krb5_rd_safe (context,
			   auth_context,
			   &packet,
			   &data,
			   NULL);
    if (status)
	krb5_err(context, 1, status, "krb5_rd_safe");

    printf ("safe packet: %.*s\n", (int)data.length,
	    (char *)data.data);

    status = krb5_read_message(context, &sock, &packet);
    if(status)
	krb5_err(context, 1, status, "krb5_read_message");

    status = krb5_rd_priv (context,
			   auth_context,
			   &packet,
			   &data,
			   NULL);
    if (status)
	krb5_err(context, 1, status, "krb5_rd_priv");

    printf ("priv packet: %.*s\n", (int)data.length,
	    (char *)data.data);

    return 0;
}

static int
doit (int port, const char *service)
{
    int sock, sock2;
    struct sockaddr_in my_addr;
    int one = 1;

    sock = socket (AF_INET, SOCK_STREAM, 0);
    if (sock < 0)
	err (1, "socket");

    memset (&my_addr, 0, sizeof(my_addr));
    my_addr.sin_family      = AF_INET;
    my_addr.sin_port        = port;
    my_addr.sin_addr.s_addr = INADDR_ANY;

    if (setsockopt (sock, SOL_SOCKET, SO_REUSEADDR,
		    (void *)&one, sizeof(one)) < 0)
	warn ("setsockopt SO_REUSEADDR");

    if (bind (sock, (struct sockaddr *)&my_addr, sizeof(my_addr)) < 0)
	err (1, "bind");

    if (listen (sock, 1) < 0)
	err (1, "listen");

    sock2 = accept (sock, NULL, NULL);
    if (sock2 < 0)
	err (1, "accept");

    return proto (sock2, service);
}

int
main(int argc, char **argv)
{
    int port = server_setup(&context, argc, argv);
    return doit (port, service_str);
}
