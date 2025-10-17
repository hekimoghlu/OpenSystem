/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 7, 2023.
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
#include <openssl/pem.h>
#include <openssl/err.h>
#include <openssl/pkcs12.h>

/* Simple PKCS#12 file creator */

int main(int argc, char **argv)
{
	FILE *fp;
	EVP_PKEY *pkey;
	X509 *cert;
	PKCS12 *p12;
	if (argc != 5) {
		fprintf(stderr, "Usage: pkwrite infile password name p12file\n");
		exit(1);
	}
	SSLeay_add_all_algorithms();
	ERR_load_crypto_strings();
	if (!(fp = fopen(argv[1], "r"))) {
		fprintf(stderr, "Error opening file %s\n", argv[1]);
		exit(1);
	}
	cert = PEM_read_X509(fp, NULL, NULL, NULL);
	rewind(fp);
	pkey = PEM_read_PrivateKey(fp, NULL, NULL, NULL);
	fclose(fp);
	p12 = PKCS12_create(argv[2], argv[3], pkey, cert, NULL, 0,0,0,0,0);
	if(!p12) {
		fprintf(stderr, "Error creating PKCS#12 structure\n");
		ERR_print_errors_fp(stderr);
		exit(1);
	}
	if (!(fp = fopen(argv[4], "wb"))) {
		fprintf(stderr, "Error opening file %s\n", argv[1]);
		ERR_print_errors_fp(stderr);
		exit(1);
	}
	i2d_PKCS12_fp(fp, p12);
	PKCS12_free(p12);
	fclose(fp);
	return 0;
}
