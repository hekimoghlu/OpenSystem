/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 11, 2023.
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
#include <config.h>
#include <err.h>
#include "getarg.h"

#include "roken.h"

#include <ifaddrs.h>

static int verbose_counter;
static int version_flag;
static int help_flag;

static struct getargs args[] = {
    {"verbose",	0,	arg_counter,	&verbose_counter,"verbose",	NULL},
    {"version",	0,	arg_flag,	&version_flag,	"print version",NULL},
    {"help",	0,	arg_flag,	&help_flag,	NULL,		NULL}
};

static void
usage(int ret)
{
    arg_printusage (args,
		    sizeof(args) / sizeof(args[0]),
		    NULL, "");
    exit (ret);
}


static void
print_addr(const char *s, struct sockaddr *sa)
{
    int i;
    printf("  %s=%d/", s, sa->sa_family);
#ifdef HAVE_STRUCT_SOCKADDR_SA_LEN
    for(i = 0; i < sa->sa_len - ((long)sa->sa_data - (long)&sa->sa_family); i++)
	printf("%02x", ((unsigned char*)sa->sa_data)[i]);
#else
    for(i = 0; i < sizeof(sa->sa_data); i++)
	printf("%02x", ((unsigned char*)sa->sa_data)[i]);
#endif
    printf("\n");
}

static void
print_ifaddrs(struct ifaddrs *x)
{
    struct ifaddrs *p;

    for(p = x; p; p = p->ifa_next) {
	if (verbose_counter) {
	    printf("%s\n", p->ifa_name);
	    printf("  flags=%x\n", p->ifa_flags);
	    if(p->ifa_addr)
		print_addr("addr", p->ifa_addr);
	    if(p->ifa_dstaddr)
		print_addr("dstaddr", p->ifa_dstaddr);
	    if(p->ifa_netmask)
		print_addr("netmask", p->ifa_netmask);
	    printf("  %p\n", p->ifa_data);
	}
    }
}

int
main(int argc, char **argv)
{
    struct ifaddrs *addrs = NULL;
    int ret, optidx = 0;

    setprogname (argv[0]);

    if (getarg (args, sizeof(args) / sizeof(args[0]), argc, argv,
		&optidx))
	usage (1);

    if (help_flag)
	usage (0);

    if (version_flag) {
	fprintf (stderr, "%s from %s-%s\n", getprogname(), PACKAGE, VERSION);
	return 0;
    }

    if (rk_SOCK_INIT())
	errx(1, "Couldn't initialize sockets. Err=%d\n", rk_SOCK_ERRNO);

    ret = getifaddrs(&addrs);
    if (ret != 0)
	err(1, "getifaddrs");

    if (addrs == NULL)
	errx(1, "address == NULL");

    print_ifaddrs(addrs);

    /* Check that freeifaddrs doesn't crash */
    freeifaddrs(addrs);

    rk_SOCK_EXIT();

    return 0;
}
