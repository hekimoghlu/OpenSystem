/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 11, 2023.
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
#include <openssl/opensslconf.h>

#ifdef OPENSSL_FIPS

static struct
    {
    unsigned char key[16];
    unsigned char plaintext[8];
    unsigned char ciphertext[8];
    } tests2[]=
	{
	{
	{ 0x7c,0x4f,0x6e,0xf7,0xa2,0x04,0x16,0xec,
	  0x0b,0x6b,0x7c,0x9e,0x5e,0x19,0xa7,0xc4 },
	{ 0x06,0xa7,0xd8,0x79,0xaa,0xce,0x69,0xef },
	{ 0x4c,0x11,0x17,0x55,0xbf,0xc4,0x4e,0xfd }
	},
	{
	{ 0x5d,0x9e,0x01,0xd3,0x25,0xc7,0x3e,0x34,
	  0x01,0x16,0x7c,0x85,0x23,0xdf,0xe0,0x68 },
	{ 0x9c,0x50,0x09,0x0f,0x5e,0x7d,0x69,0x7e },
	{ 0xd2,0x0b,0x18,0xdf,0xd9,0x0d,0x9e,0xff },
	}
	};

static struct
    {
    unsigned char key[24];
    unsigned char plaintext[8];
    unsigned char ciphertext[8];
    } tests3[]=
	{
	{
	{ 0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
	  0xFE,0xDC,0xBA,0x98,0x76,0x54,0x32,0x10,
	  0x12,0x34,0x56,0x78,0x9a,0xbc,0xde,0xf0 },
	{ 0x8f,0x8f,0xbf,0x9b,0x5d,0x48,0xb4,0x1c },
	{ 0x59,0x8c,0xe5,0xd3,0x6c,0xa2,0xea,0x1b },
	},
	{
	{ 0xDC,0xBA,0x98,0x76,0x54,0x32,0x10,0xFE,
	  0x01,0x23,0x45,0x67,0x89,0xAB,0xCD,0xEF,
	  0xED,0x39,0xD9,0x50,0xFA,0x74,0xBC,0xC4 },
	{ 0x01,0x23,0x45,0x67,0x89,0xAB,0xCD,0xEF },
	{ 0x11,0x25,0xb0,0x35,0xbe,0xa0,0x82,0x86 },
	},
	};

void FIPS_corrupt_des()
    {
    tests2[0].plaintext[0]++;
    }

int FIPS_selftest_des()
    {
    int n, ret = 0;
    EVP_CIPHER_CTX ctx;
    EVP_CIPHER_CTX_init(&ctx);
    /* Encrypt/decrypt with 2-key 3DES and compare to known answers */
    for(n=0 ; n < 2 ; ++n)
	{
	if (!fips_cipher_test(&ctx, EVP_des_ede_ecb(),
				tests2[n].key, NULL,
				tests2[n].plaintext, tests2[n].ciphertext, 8))
		goto err;
	}

    /* Encrypt/decrypt with 3DES and compare to known answers */
    for(n=0 ; n < 2 ; ++n)
	{
	if (!fips_cipher_test(&ctx, EVP_des_ede3_ecb(),
				tests3[n].key, NULL,
				tests3[n].plaintext, tests3[n].ciphertext, 8))
		goto err;
	}
    ret = 1;
    err:
    EVP_CIPHER_CTX_cleanup(&ctx);
    if (ret == 0)
	    FIPSerr(FIPS_F_FIPS_SELFTEST_DES,FIPS_R_SELFTEST_FAILED);

    return ret;
    }
#endif
