/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 21, 2024.
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <openssl/opensslconf.h>
#include <openssl/sha.h>
#include <openssl/hmac.h>

#ifndef FIPSCANISTER_O
int FIPS_selftest_failed() { return 0; }
void FIPS_selftest_check() {}
void OPENSSL_cleanse(void *p,size_t len) {}
#endif

#ifdef OPENSSL_FIPS

static void hmac_init(SHA_CTX *md_ctx,SHA_CTX *o_ctx,
		      const char *key)
    {
    size_t len=strlen(key);
    int i;
    unsigned char keymd[HMAC_MAX_MD_CBLOCK];
    unsigned char pad[HMAC_MAX_MD_CBLOCK];

    if (len > SHA_CBLOCK)
	{
	SHA1_Init(md_ctx);
	SHA1_Update(md_ctx,key,len);
	SHA1_Final(keymd,md_ctx);
	len=20;
	}
    else
	memcpy(keymd,key,len);
    memset(&keymd[len],'\0',HMAC_MAX_MD_CBLOCK-len);

    for(i=0 ; i < HMAC_MAX_MD_CBLOCK ; i++)
	pad[i]=0x36^keymd[i];
    SHA1_Init(md_ctx);
    SHA1_Update(md_ctx,pad,SHA_CBLOCK);

    for(i=0 ; i < HMAC_MAX_MD_CBLOCK ; i++)
	pad[i]=0x5c^keymd[i];
    SHA1_Init(o_ctx);
    SHA1_Update(o_ctx,pad,SHA_CBLOCK);
    }

static void hmac_final(unsigned char *md,SHA_CTX *md_ctx,SHA_CTX *o_ctx)
    {
    unsigned char buf[20];

    SHA1_Final(buf,md_ctx);
    SHA1_Update(o_ctx,buf,sizeof buf);
    SHA1_Final(md,o_ctx);
    }

#endif

int main(int argc,char **argv)
    {
#ifdef OPENSSL_FIPS
    static char key[]="etaonrishdlcupfm";
    int n,binary=0;

    if(argc < 2)
	{
	fprintf(stderr,"%s [<file>]+\n",argv[0]);
	exit(1);
	}

    n=1;
    if (!strcmp(argv[n],"-binary"))
	{
	n++;
	binary=1;	/* emit binary fingerprint... */
	}

    for(; n < argc ; ++n)
	{
	FILE *f=fopen(argv[n],"rb");
	SHA_CTX md_ctx,o_ctx;
	unsigned char md[20];
	int i;

	if(!f)
	    {
	    perror(argv[n]);
	    exit(2);
	    }

	hmac_init(&md_ctx,&o_ctx,key);
	for( ; ; )
	    {
	    char buf[1024];
	    size_t l=fread(buf,1,sizeof buf,f);

	    if(l == 0)
		{
		if(ferror(f))
		    {
		    perror(argv[n]);
		    exit(3);
		    }
		else
		    break;
		}
	    SHA1_Update(&md_ctx,buf,l);
	    }
	hmac_final(md,&md_ctx,&o_ctx);

	if (binary)
	    {
	    fwrite(md,20,1,stdout);
	    break;	/* ... for single(!) file */
	    }

	printf("HMAC-SHA1(%s)= ",argv[n]);
	for(i=0 ; i < 20 ; ++i)
	    printf("%02x",md[i]);
	printf("\n");
	}
#endif
    return 0;
    }


