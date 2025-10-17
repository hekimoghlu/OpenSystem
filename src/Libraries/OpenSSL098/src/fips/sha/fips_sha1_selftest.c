/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 6, 2021.
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
#include <openssl/err.h>
#include <openssl/fips.h>
#include <openssl/evp.h>
#include <openssl/sha.h>

#ifdef OPENSSL_FIPS
static char test[][60]=
    {
    "",
    "abc",
    "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
    };

static const unsigned char ret[][SHA_DIGEST_LENGTH]=
    {
    { 0xda,0x39,0xa3,0xee,0x5e,0x6b,0x4b,0x0d,0x32,0x55,
      0xbf,0xef,0x95,0x60,0x18,0x90,0xaf,0xd8,0x07,0x09 },
    { 0xa9,0x99,0x3e,0x36,0x47,0x06,0x81,0x6a,0xba,0x3e,
      0x25,0x71,0x78,0x50,0xc2,0x6c,0x9c,0xd0,0xd8,0x9d },
    { 0x84,0x98,0x3e,0x44,0x1c,0x3b,0xd2,0x6e,0xba,0xae,
      0x4a,0xa1,0xf9,0x51,0x29,0xe5,0xe5,0x46,0x70,0xf1 },
    };

void FIPS_corrupt_sha1()
    {
    test[2][0]++;
    }

int FIPS_selftest_sha1()
    {
    size_t n;

    for(n=0 ; n<sizeof(test)/sizeof(test[0]) ; ++n)
	{
	unsigned char md[SHA_DIGEST_LENGTH];

	EVP_Digest(test[n],strlen(test[n]),md, NULL, EVP_sha1(), NULL);
	if(memcmp(md,ret[n],sizeof md))
	    {
	    FIPSerr(FIPS_F_FIPS_SELFTEST_SHA1,FIPS_R_SELFTEST_FAILED);
	    return 0;
	    }
	}
    return 1;
    }

#endif
