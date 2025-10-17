/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 27, 2025.
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

#include "roken.h"
#include "getarg.h"
#ifdef HAVE_ARPA_NAMESER_H
#include <arpa/nameser.h>
#endif
#ifdef HAVE_RESOLV_H
#include <resolv.h>
#endif
#include "resolve.h"

static int loop_integer = 1;
static int version_flag = 0;
static int help_flag	= 0;

static struct getargs args[] = {
    {"loop",	0,	arg_integer,	&loop_integer,
     "loop resolving", NULL },
    {"version",	0,	arg_flag,	&version_flag,
     "print version", NULL },
    {"help",	0,	arg_flag,	&help_flag,
     NULL, NULL }
};

static void
usage (int ret)
{
    arg_printusage (args,
		    sizeof(args)/sizeof(*args),
		    NULL,
		    "dns-record resource-record-type");
    exit (ret);
}

int
main(int argc, char **argv)
{
    struct rk_dns_reply *r;
    struct rk_resource_record *rr;
    int optidx = 0, i, exit_code = 0;

    setprogname (argv[0]);

    if(getarg(args, sizeof(args) / sizeof(args[0]), argc, argv, &optidx))
	usage(1);

    if (help_flag)
	usage (0);

    if(version_flag){
	printf("some version\n");
	exit(0);
    }

    argc -= optidx;
    argv += optidx;

    if (argc != 2)
	usage(1);

    for (i = 0; i < loop_integer; i++) {

	r = rk_dns_lookup(argv[0], argv[1]);
	if(r == NULL){
	    printf("No reply.\n");
	    exit_code = 1;
	    break;
	}
	if(r->q.type == rk_ns_t_srv)
	    rk_dns_srv_order(r);

	for(rr = r->head; rr;rr=rr->next){
	    printf("%-30s %-5s %-6d ", rr->domain, rk_dns_type_to_string(rr->type), rr->ttl);
	    switch(rr->type){
	    case rk_ns_t_ns:
	    case rk_ns_t_cname:
	    case rk_ns_t_ptr:
		printf("%s\n", (char*)rr->u.data);
		break;
	    case rk_ns_t_a:
		printf("%s\n", inet_ntoa(*rr->u.a));
		break;
	    case rk_ns_t_mx:
	    case rk_ns_t_afsdb:{
		printf("%d %s\n", rr->u.mx->preference, rr->u.mx->domain);
		break;
	    }
	    case rk_ns_t_srv:{
		struct rk_srv_record *srv = rr->u.srv;
		printf("%d %d %d %s\n", srv->priority, srv->weight,
		       srv->port, srv->target);
		break;
	    }
	    case rk_ns_t_txt: {
		printf("%s\n", rr->u.txt);
		break;
	    }
	    case rk_ns_t_sig : {
		struct rk_sig_record *sig = rr->u.sig;
		const char *type_string = rk_dns_type_to_string (sig->type);

		printf ("type %u (%s), algorithm %u, labels %u, orig_ttl %u, sig_expiration %u, sig_inception %u, key_tag %u, signer %s\n",
			sig->type, type_string ? type_string : "",
			sig->algorithm, sig->labels, sig->orig_ttl,
			sig->sig_expiration, sig->sig_inception, sig->key_tag,
			sig->signer);
		break;
	    }
	    case rk_ns_t_key : {
		struct rk_key_record *key = rr->u.key;

		printf ("flags %u, protocol %u, algorithm %u\n",
			key->flags, key->protocol, key->algorithm);
		break;
	    }
	    case rk_ns_t_sshfp : {
		struct rk_sshfp_record *sshfp = rr->u.sshfp;
		size_t i;

		printf ("alg %u type %u length %lu data ", sshfp->algorithm,
			sshfp->type,  (unsigned long)sshfp->sshfp_len);
		for (i = 0; i < sshfp->sshfp_len; i++)
		    printf("%02X", sshfp->sshfp_data[i]);
		printf("\n");

		break;
	    }
	    case rk_ns_t_ds : {
		struct rk_ds_record *ds = rr->u.ds;
		size_t i;

		printf ("key tag %u alg %u type %u length %lu data ",
			ds->key_tag, ds->algorithm, ds->digest_type,
			(unsigned long)ds->digest_len);
		for (i = 0; i < ds->digest_len; i++)
		    printf("%02X", ds->digest_data[i]);
		printf("\n");

		break;
	    }
	    default:
		printf("\n");
		break;
	    }
	}
	rk_dns_free_data(r);
    }

    return exit_code;
}
