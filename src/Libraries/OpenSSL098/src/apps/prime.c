/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 10, 2021.
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
#include <string.h>

#include "apps.h"
#include <openssl/bn.h>


#undef PROG
#define PROG prime_main

int MAIN(int, char **);

int MAIN(int argc, char **argv)
    {
    int hex=0;
    int checks=20;
    BIGNUM *bn=NULL;
    BIO *bio_out;

    apps_startup();

    if (bio_err == NULL)
	if ((bio_err=BIO_new(BIO_s_file())) != NULL)
	    BIO_set_fp(bio_err,stderr,BIO_NOCLOSE|BIO_FP_TEXT);

    --argc;
    ++argv;
    while (argc >= 1 && **argv == '-')
	{
	if(!strcmp(*argv,"-hex"))
	    hex=1;
	else if(!strcmp(*argv,"-checks"))
	    if(--argc < 1)
		goto bad;
	    else
		checks=atoi(*++argv);
	else
	    {
	    BIO_printf(bio_err,"Unknown option '%s'\n",*argv);
	    goto bad;
	    }
	--argc;
	++argv;
	}

    if (argv[0] == NULL)
	{
	BIO_printf(bio_err,"No prime specified\n");
	goto bad;
	}

   if ((bio_out=BIO_new(BIO_s_file())) != NULL)
	{
	BIO_set_fp(bio_out,stdout,BIO_NOCLOSE);
#ifdef OPENSSL_SYS_VMS
	    {
	    BIO *tmpbio = BIO_new(BIO_f_linebuffer());
	    bio_out = BIO_push(tmpbio, bio_out);
	    }
#endif
	}

    if(hex)
	BN_hex2bn(&bn,argv[0]);
    else
	BN_dec2bn(&bn,argv[0]);

    BN_print(bio_out,bn);
    BIO_printf(bio_out," is %sprime\n",
	       BN_is_prime_ex(bn,checks,NULL,NULL) ? "" : "not ");

    BN_free(bn);
    BIO_free_all(bio_out);

    return 0;

    bad:
    BIO_printf(bio_err,"options are\n");
    BIO_printf(bio_err,"%-14s hex\n","-hex");
    BIO_printf(bio_err,"%-14s number of checks\n","-checks <n>");
    return 1;
    }
