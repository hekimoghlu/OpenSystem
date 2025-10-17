/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 26, 2022.
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

/* Simple PKCS#12 file reader */

int main(int argc, char **argv)
{
	FILE *fp;
	EVP_PKEY *pkey;
	X509 *cert;
	STACK_OF(X509) *ca = NULL;
	PKCS12 *p12;
	int i;
	if (argc != 4) {
		fprintf(stderr, "Usage: pkread p12file password opfile\n");
		exit (1);
	}
	SSLeay_add_all_algorithms();
	ERR_load_crypto_strings();
	if (!(fp = fopen(argv[1], "rb"))) {
		fprintf(stderr, "Error opening file %s\n", argv[1]);
		exit(1);
	}
	p12 = d2i_PKCS12_fp(fp, NULL);
	fclose (fp);
	if (!p12) {
		fprintf(stderr, "Error reading PKCS#12 file\n");
		ERR_print_errors_fp(stderr);
		exit (1);
	}
	if (!PKCS12_parse(p12, argv[2], &pkey, &cert, &ca)) {
		fprintf(stderr, "Error parsing PKCS#12 file\n");
		ERR_print_errors_fp(stderr);
		exit (1);
	}
	PKCS12_free(p12);
	if (!(fp = fopen(argv[3], "w"))) {
		fprintf(stderr, "Error opening file %s\n", argv[1]);
		exit(1);
	}
	if (pkey) {
		fprintf(fp, "***Private Key***\n");
		PEM_write_PrivateKey(fp, pkey, NULL, NULL, 0, NULL, NULL);
	}
	if (cert) {
		fprintf(fp, "***User Certificate***\n");
		PEM_write_X509_AUX(fp, cert);
	}
	if (ca && sk_num(ca)) {
		fprintf(fp, "***Other Certificates***\n");
		for (i = 0; i < sk_X509_num(ca); i++) 
		    PEM_write_X509_AUX(fp, sk_X509_value(ca, i));
	}
	fclose(fp);
	return 0;
}
