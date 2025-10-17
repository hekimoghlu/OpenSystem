/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 7, 2022.
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

#ifdef OPENSSL_FIPS
static struct
    {
    unsigned char key[16];
    unsigned char plaintext[16];
    unsigned char ciphertext[16];
    } tests[]=
	{
	{
	{ 0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
	  0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F },
	{ 0x00,0x11,0x22,0x33,0x44,0x55,0x66,0x77,
	  0x88,0x99,0xAA,0xBB,0xCC,0xDD,0xEE,0xFF },
	{ 0x69,0xC4,0xE0,0xD8,0x6A,0x7B,0x04,0x30,
	  0xD8,0xCD,0xB7,0x80,0x70,0xB4,0xC5,0x5A },
	},
	};

void FIPS_corrupt_aes()
    {
    tests[0].key[0]++;
    }

int FIPS_selftest_aes()
    {
    int n;
    int ret = 0;
    EVP_CIPHER_CTX ctx;
    EVP_CIPHER_CTX_init(&ctx);

    for(n=0 ; n < 1 ; ++n)
	{
	if (fips_cipher_test(&ctx, EVP_aes_128_ecb(),
				tests[n].key, NULL,
				tests[n].plaintext,
				tests[n].ciphertext,
				16) <= 0)
		goto err;
	}
    ret = 1;
    err:
    EVP_CIPHER_CTX_cleanup(&ctx);
    if (ret == 0)
	    FIPSerr(FIPS_F_FIPS_SELFTEST_AES,FIPS_R_SELFTEST_FAILED);
    return ret;
    }
#endif
