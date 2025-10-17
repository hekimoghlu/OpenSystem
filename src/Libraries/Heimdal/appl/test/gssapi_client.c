/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 28, 2024.
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
#include <gssapi/gssapi.h>
#include <gssapi/gssapi_krb5.h>
#include <gssapi/gssapi_spnego.h>
#include "gss_common.h"
RCSID("$Id$");

static int
do_trans (int sock, gss_ctx_id_t context_hdl)
{
    OM_uint32 maj_stat, min_stat;
    gss_buffer_desc real_input_token, real_output_token;
    gss_buffer_t input_token = &real_input_token,
	output_token = &real_output_token;
    int conf_flag;

    /* get_mic */

    input_token->length = 3;
    input_token->value  = strdup("hej");

    maj_stat = gss_get_mic(&min_stat,
			   context_hdl,
			   GSS_C_QOP_DEFAULT,
			   input_token,
			   output_token);
    if (GSS_ERROR(maj_stat))
	gss_err (1, min_stat, "gss_get_mic");

    write_token (sock, input_token);
    write_token (sock, output_token);

    gss_release_buffer(&min_stat, output_token);

    /* verify mic */

    read_token (sock, input_token);
    read_token (sock, output_token);

    maj_stat = gss_verify_mic(&min_stat,
			      context_hdl,
			      input_token,
			      output_token,
			      NULL);
    if (GSS_ERROR(maj_stat))
	gss_err (1, min_stat, "gss_verify_mic");

    gss_release_buffer (&min_stat, input_token);
    gss_release_buffer (&min_stat, output_token);

    /* wrap */

    input_token->length = 7;
    input_token->value  = "hemligt";

    maj_stat = gss_wrap (&min_stat,
			 context_hdl,
			 0,
			 GSS_C_QOP_DEFAULT,
			 input_token,
			 NULL,
			 output_token);
    if (GSS_ERROR(maj_stat))
	gss_err (1, min_stat, "gss_wrap");

    write_token (sock, output_token);

    maj_stat = gss_wrap (&min_stat,
			 context_hdl,
			 1,
			 GSS_C_QOP_DEFAULT,
			 input_token,
			 NULL,
			 output_token);
    if (GSS_ERROR(maj_stat))
	gss_err (1, min_stat, "gss_wrap");

    write_token (sock, output_token);

    read_token (sock, input_token);

    maj_stat = gss_unwrap (&min_stat,
			   context_hdl,
			   input_token,
			   output_token,
			   &conf_flag,
			   NULL);
    if(GSS_ERROR(maj_stat))
	gss_err (1, min_stat, "gss_unwrap");
	
    write_token (sock, output_token);

    gss_release_buffer(&min_stat, output_token);

    return 0;
}

extern char *password;

static int
proto (int sock, const char *hostname, const char *service)
{
    struct sockaddr_storage remote, local;
    socklen_t addrlen;

    int context_established = 0;
    gss_ctx_id_t context_hdl = GSS_C_NO_CONTEXT;
    gss_cred_id_t cred = GSS_C_NO_CREDENTIAL;
    gss_buffer_desc real_input_token, real_output_token;
    gss_buffer_t input_token = &real_input_token,
	output_token = &real_output_token;
    OM_uint32 maj_stat, min_stat;
    gss_name_t server;
    gss_buffer_desc name_token;
    u_char init_buf[4];
    u_char acct_buf[4];
    gss_OID mech_oid;
    char *str;

    mech_oid = select_mech(mech);

    name_token.length = asprintf (&str,
				  "%s@%s", service, hostname);
    if (str == NULL)
	errx(1, "malloc - out of memory");
    name_token.value = str;

    maj_stat = gss_import_name (&min_stat,
				&name_token,
				GSS_C_NT_HOSTBASED_SERVICE,
				&server);
    if (GSS_ERROR(maj_stat))
	gss_err (1, min_stat,
		 "Error importing name `%s@%s':\n", service, hostname);

    if (password) {
        gss_buffer_desc pw;

        pw.value = password;
        pw.length = strlen(password);

        maj_stat = gss_acquire_cred_with_password(&min_stat,
						  GSS_C_NO_NAME,
						  &pw,
						  GSS_C_INDEFINITE,
						  GSS_C_NO_OID_SET,
						  GSS_C_INITIATE,
						  &cred,
						  NULL,
						  NULL);
        if (GSS_ERROR(maj_stat))
            gss_err (1, min_stat,
                     "Error acquiring default initiator credentials");
    }

    addrlen = sizeof(local);
    if (getsockname (sock, (struct sockaddr *)&local, &addrlen) < 0
	|| addrlen > sizeof(local))
	err (1, "getsockname(%s)", hostname);

    addrlen = sizeof(remote);
    if (getpeername (sock, (struct sockaddr *)&remote, &addrlen) < 0
	|| addrlen > sizeof(remote))
	err (1, "getpeername(%s)", hostname);

    input_token->length = 0;
    output_token->length = 0;

#if 0
    struct gss_channel_bindings_struct input_chan_bindings;

    input_chan_bindings.initiator_addrtype = GSS_C_AF_INET;
    input_chan_bindings.initiator_address.length = 4;
    init_buf[0] = (local.sin_addr.s_addr >> 24) & 0xFF;
    init_buf[1] = (local.sin_addr.s_addr >> 16) & 0xFF;
    init_buf[2] = (local.sin_addr.s_addr >>  8) & 0xFF;
    init_buf[3] = (local.sin_addr.s_addr >>  0) & 0xFF;
    input_chan_bindings.initiator_address.value = init_buf;

    input_chan_bindings.acceptor_addrtype = GSS_C_AF_INET;
    input_chan_bindings.acceptor_address.length = 4;
    acct_buf[0] = (remote.sin_addr.s_addr >> 24) & 0xFF;
    acct_buf[1] = (remote.sin_addr.s_addr >> 16) & 0xFF;
    acct_buf[2] = (remote.sin_addr.s_addr >>  8) & 0xFF;
    acct_buf[3] = (remote.sin_addr.s_addr >>  0) & 0xFF;
    input_chan_bindings.acceptor_address.value = acct_buf;

    input_chan_bindings.application_data.value = emalloc(4);
    * (unsigned short*)input_chan_bindings.application_data.value = local.sin_port;
    * ((unsigned short *)input_chan_bindings.application_data.value + 1) = remote.sin_port;
    input_chan_bindings.application_data.length = 4;

    input_chan_bindings.application_data.length = 0;
    input_chan_bindings.application_data.value = NULL;
#endif

    while(!context_established) {
	maj_stat =
	    gss_init_sec_context(&min_stat,
				 cred,
				 &context_hdl,
				 server,
				 mech_oid,
				 GSS_C_MUTUAL_FLAG | GSS_C_SEQUENCE_FLAG,
				 0,
				 NULL,
				 input_token,
				 NULL,
				 output_token,
				 NULL,
				 NULL);
	if (GSS_ERROR(maj_stat))
	    gss_err (1, min_stat, "gss_init_sec_context");
	if (output_token->length != 0)
	    write_token (sock, output_token);
	if (GSS_ERROR(maj_stat)) {
	    if (context_hdl != GSS_C_NO_CONTEXT)
		gss_delete_sec_context (&min_stat,
					&context_hdl,
					GSS_C_NO_BUFFER);
	    break;
	}
	if (maj_stat & GSS_S_CONTINUE_NEEDED) {
	    read_token (sock, input_token);
	} else {
	    context_established = 1;
	}

    }
    if (fork_flag) {
	pid_t pid;
	int pipefd[2];

	if (pipe (pipefd) < 0)
	    err (1, "pipe");

	pid = fork ();
	if (pid < 0)
	    err (1, "fork");
	if (pid != 0) {
	    gss_buffer_desc buf;

	    maj_stat = gss_export_sec_context (&min_stat,
					       &context_hdl,
					       &buf);
	    if (GSS_ERROR(maj_stat))
		gss_err (1, min_stat, "gss_export_sec_context");
	    write_token (pipefd[1], &buf);
	    exit (0);
	} else {
	    gss_ctx_id_t context_hdl;
	    gss_buffer_desc buf;

	    close (pipefd[1]);
	    read_token (pipefd[0], &buf);
	    close (pipefd[0]);
	    maj_stat = gss_import_sec_context (&min_stat, &buf, &context_hdl);
	    if (GSS_ERROR(maj_stat))
		gss_err (1, min_stat, "gss_import_sec_context");
	    gss_release_buffer (&min_stat, &buf);
	    return do_trans (sock, context_hdl);
	}
    } else {
	return do_trans (sock, context_hdl);
    }
}

int
main(int argc, char **argv)
{
    krb5_context context; /* XXX */
    int port = client_setup(&context, &argc, argv);
    return client_doit (argv[argc], port, service, proto);
}
