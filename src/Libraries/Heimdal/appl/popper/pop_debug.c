/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 19, 2022.
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
/* Tiny program to help debug popper */

#include "popper.h"
RCSID("$Id$");

static void
loop(int s)
{
    char cmd[1024];
    char buf[1024];
    fd_set fds;
    while(1){
	FD_ZERO(&fds);
	FD_SET(0, &fds);
	FD_SET(s, &fds);
	if(select(s+1, &fds, 0, 0, 0) < 0)
	    err(1, "select");
	if(FD_ISSET(0, &fds)){
	    fgets(cmd, sizeof(cmd), stdin);
	    cmd[strlen(cmd) - 1] = '\0';
	    strlcat (cmd, "\r\n", sizeof(cmd));
	    write(s, cmd, strlen(cmd));
	}
	if(FD_ISSET(s, &fds)){
	    int n = read(s, buf, sizeof(buf));
	    if(n == 0)
		exit(0);
	    fwrite(buf, n, 1, stdout);
	}
    }
}

static int
get_socket (const char *hostname, int port)
{
    int ret;
    struct addrinfo *ai, *a;
    struct addrinfo hints;
    char portstr[NI_MAXSERV];

    memset (&hints, 0, sizeof(hints));
    hints.ai_socktype = SOCK_STREAM;
    snprintf (portstr, sizeof(portstr), "%d", ntohs(port));
    ret = getaddrinfo (hostname, portstr, &hints, &ai);
    if (ret)
	errx (1, "getaddrinfo %s: %s", hostname, gai_strerror (ret));

    for (a = ai; a != NULL; a = a->ai_next) {
	int s;

	s = socket (a->ai_family, a->ai_socktype, a->ai_protocol);
	if (s < 0)
	    continue;
	if (connect (s, a->ai_addr, a->ai_addrlen) < 0) {
	    close (s);
	    continue;
	}
	freeaddrinfo (ai);
	return s;
    }
    err (1, "failed to connect to %s", hostname);
}

#ifdef KRB5
static int
doit_v5 (char *host, int port)
{
    krb5_error_code ret;
    krb5_context context;
    krb5_auth_context auth_context = NULL;
    krb5_principal server;
    int s = get_socket (host, port);

    ret = krb5_init_context (&context);
    if (ret)
	errx (1, "krb5_init_context failed: %d", ret);

    ret = krb5_sname_to_principal (context,
				   host,
				   "pop",
				   KRB5_NT_SRV_HST,
				   &server);
    if (ret) {
	warnx ("krb5_sname_to_principal: %s",
	       krb5_get_err_text (context, ret));
	return 1;
    }
    ret = krb5_sendauth (context,
			 &auth_context,
			 &s,
			 "KPOPV1.0",
			 NULL,
			 server,
			 0,
			 NULL,
			 NULL,
			 NULL,
			 NULL,
			 NULL,
			 NULL);
     if (ret) {
	 warnx ("krb5_sendauth: %s",
		krb5_get_err_text (context, ret));
	 return 1;
     }
     loop (s);
     return 0;
}
#endif


#ifdef KRB5
static int use_v5 = -1;
#endif
static char *port_str;
static int do_version;
static int do_help;

struct getargs args[] = {
#ifdef KRB5
    { "krb5",	'5', arg_flag,		&use_v5,	"Use Kerberos V5",
      NULL },
#endif
    { "port",	'p', arg_string,	&port_str,	"Use this port",
      "number-or-service" },
    { "version", 0,  arg_flag,		&do_version,	"Print version",
      NULL },
    { "help",	 0,  arg_flag,		&do_help,	NULL,
      NULL }
};

static void
usage (int ret)
{
    arg_printusage (args,
		    sizeof(args) / sizeof(args[0]),
		    NULL,
		    "hostname");
    exit (ret);
}

int
main(int argc, char **argv)
{
    int port = 0;
    int ret = 1;
    int optind = 0;

    setprogname(argv[0]);

    if (getarg (args, sizeof(args) / sizeof(args[0]), argc, argv,
		&optind))
	usage (1);

    argc -= optind;
    argv += optind;

    if (do_help)
	usage (0);

    if (do_version) {
	print_version (NULL);
	return 0;
    }

    if (argc < 1)
	usage (1);

    if (port_str) {
	struct servent *s = roken_getservbyname (port_str, "tcp");

	if (s)
	    port = s->s_port;
	else {
	    char *ptr;

	    port = strtol (port_str, &ptr, 10);
	    if (port == 0 && ptr == port_str)
		errx (1, "Bad port `%s'", port_str);
	    port = htons(port);
	}
    }
    if (port == 0) {
#ifdef KRB5
	port = krb5_getportbyname (NULL, "kpop", "tcp", 1109);
#else
#error must define KRB5
#endif
    }

#ifdef KRB5
    if (ret && use_v5) {
	ret = doit_v5 (argv[0], port);
    }
#endif
    return ret;
}
