/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 11, 2021.
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
#include <krb5-types.h> /* should really be stdint.h */
#include <hcrypto/evp.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <err.h>
#include <assert.h>

#include "roken.h"

/* key and initial vector */
static char key[16] =
    "\xaa\xbb\x45\xd4\xaa\xbb\x45\xd4"
    "\xaa\xbb\x45\xd4\xaa\xbb\x45\xd4";
static char ivec[16] =
    "\xaa\xbb\x45\xd4\xaa\xbb\x45\xd4"
    "\xaa\xbb\x45\xd4\xaa\xbb\x45\xd4";

static void
usage(int exit_code) __attribute__((noreturn));

static void
usage(int exit_code)
{
    printf("usage: %s in out\n", getprogname());
    exit(exit_code);
}


int
main(int argc, char **argv)
{
    int encryptp = 1;
    const char *ifn = NULL, *ofn = NULL;
    FILE *in, *out;
    void *ibuf, *obuf;
    int ilen, olen;
    size_t block_size = 0;
    const EVP_CIPHER *c = EVP_aes_128_cbc();
    EVP_CIPHER_CTX ctx;
    int ret;

    setprogname(argv[0]);

    if (argc == 2) {
	if (strcmp(argv[1], "--version") == 0) {
	    printf("version");
	    exit(0);
	}
	if (strcmp(argv[1], "--help") == 0)
	    usage(0);
	usage(1);
    } else if (argc == 4) {
	block_size = atoi(argv[1]);
	if (block_size == 0)
	    errx(1, "invalid blocksize %s", argv[1]);
	ifn = argv[2];
	ofn = argv[3];
    } else
	usage(1);

    in = fopen(ifn, "r");
    if (in == NULL)
	errx(1, "failed to open input file");
    out = fopen(ofn, "w+");
    if (out == NULL)
	errx(1, "failed to open output file");

    /* Check that key and ivec are long enough */
    assert(EVP_CIPHER_key_length(c) <= sizeof(key));
    assert(EVP_CIPHER_iv_length(c) <= sizeof(ivec));

    /*
     * Allocate buffer, the output buffer is at least
     * EVP_CIPHER_block_size() longer
     */
    ibuf = malloc(block_size);
    obuf = malloc(block_size + EVP_CIPHER_block_size(c));

    /*
     * Init the memory used for EVP_CIPHER_CTX and set the key and
     * ivec.
     */
    EVP_CIPHER_CTX_init(&ctx);
    EVP_CipherInit_ex(&ctx, c, NULL, key, ivec, encryptp);

    /* read in buffer */
    while ((ilen = fread(ibuf, 1, block_size, in)) > 0) {
	/* encrypto/decrypt */
	ret = EVP_CipherUpdate(&ctx, obuf, &olen, ibuf, ilen);
	if (ret != 1) {
	    EVP_CIPHER_CTX_cleanup(&ctx);
	    errx(1, "EVP_CipherUpdate failed");
	}
	/* write out to output file */
	fwrite(obuf, 1, olen, out);
    }
    /* done reading */
    fclose(in);

    /* clear up any last bytes left in the output buffer */
    ret = EVP_CipherFinal_ex(&ctx, obuf, &olen);
    EVP_CIPHER_CTX_cleanup(&ctx);
    if (ret != 1)
	errx(1, "EVP_CipherFinal_ex failed");

    /* write the last bytes out and close */
    fwrite(obuf, 1, olen, out);
    fclose(out);

    return 0;
}
