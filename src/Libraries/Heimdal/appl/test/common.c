/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 19, 2025.
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

static int help_flag;
static int version_flag;
static char *port_str;
char *keytab_str;
krb5_keytab keytab;
char *service_str = SERVICE;
char *mech = "krb5";
int fork_flag;
char *password = NULL;

static struct getargs args[] = {
    { "port", 'p', arg_string, &port_str, "port to listen to", "port" },
    { "service", 's', arg_string, &service_str, "service to use", "service" },
    { "keytab", 'k', arg_string, &keytab_str, "keytab to use", "keytab" },
    { "mech", 'm', arg_string, &mech, "gssapi mech to use", "mech" },
    { "password", 'P', arg_string, &password, "password to use", "password" },
    { "fork", 'f', arg_flag, &fork_flag, "do fork" },
    { "help", 'h', arg_flag, &help_flag },
    { "version", 0, arg_flag, &version_flag }
};

static int num_args = sizeof(args) / sizeof(args[0]);

static void
server_usage(int code, struct getargs *largs, int lnum_args)
{
    arg_printusage(largs, lnum_args, NULL, "");
    exit(code);
}

static void
client_usage(int code, struct getargs *largs, int lnum_args)
{
    arg_printusage(largs, lnum_args, NULL, "host");
    exit(code);
}


static int
common_setup(krb5_context *context, int *argc, char **argv,
	     void (*usage)(int, struct getargs*, int))
{
    int port = 0;
    *argc = krb5_program_setup(context, *argc, argv, args, num_args, usage);

    if(help_flag)
	(*usage)(0, args, num_args);
    if(version_flag) {
	print_version(NULL);
	exit(0);
    }

    if(port_str){
	struct servent *s = roken_getservbyname(port_str, "tcp");
	if(s)
	    port = s->s_port;
	else {
	    char *ptr;

	    port = (int)strtol (port_str, &ptr, 10);
	    if (port == 0 && ptr == port_str)
		errx (1, "Bad port `%s'", port_str);
	    port = htons(port);
	}
    }

    if (port == 0)
	port = krb5_getportbyname (*context, PORT, "tcp", 4711);

    return port;
}

int
server_setup(krb5_context *context, int argc, char **argv)
{
    int port = common_setup(context, &argc, argv, server_usage);
    krb5_error_code ret;

    if(argv[argc] != NULL)
	server_usage(1, args, num_args);
    if (keytab_str != NULL)
	ret = krb5_kt_resolve (*context, keytab_str, &keytab);
    else
	ret = krb5_kt_default (*context, &keytab);
    if (ret)
	krb5_err (*context, 1, ret, "krb5_kt_resolve/default");
    return port;
}

int
client_setup(krb5_context *context, int *argc, char **argv)
{
    int optidx = *argc;
    int port = common_setup(context, &optidx, argv, client_usage);
    if(*argc - optidx != 1)
	client_usage(1, args, num_args);
    *argc = optidx;
    return port;
}

int
client_doit (const char *hostname, int port, const char *serv,
	     int (*func)(int, const char *hostname, const char *service))
{
    struct addrinfo *ai, *a;
    struct addrinfo hints;
    int error;
    char portstr[NI_MAXSERV];

    memset (&hints, 0, sizeof(hints));
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;

    snprintf (portstr, sizeof(portstr), "%u", ntohs(port));

    error = getaddrinfo (hostname, portstr, &hints, &ai);
    if (error) {
	errx (1, "%s: %s", hostname, gai_strerror(error));
    }

    for (a = ai; a != NULL; a = a->ai_next) {
	int s;

	s = socket (a->ai_family, a->ai_socktype, a->ai_protocol);
	if (s < 0)
	    continue;
	if (connect (s, a->ai_addr, a->ai_addrlen) < 0) {
	    warn ("connect(%s)", hostname);
	    close (s);
	    continue;
	}
	freeaddrinfo (ai);
	return (*func) (s, hostname, serv);
    }
    warnx ("failed to contact %s", hostname);
    freeaddrinfo (ai);
    return 1;
}
