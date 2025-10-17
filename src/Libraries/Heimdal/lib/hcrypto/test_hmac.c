/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 13, 2022.
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

#include <sys/types.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <hmac.h>
#include <evp.h>
#include <roken.h>

int
main(int argc, char **argv)
{
    unsigned char buf[4] = { 0, 0, 0, 0 };
    char hmackey[] = "hello-world";
    size_t hmackey_size = sizeof(hmackey);
    unsigned int hmaclen;
    unsigned char hmac[EVP_MAX_MD_SIZE];
    HMAC_CTX c;

    char answer[20] = "\x2c\xfa\x32\xb7\x2b\x8a\xf6\xdf\xcf\xda"
	              "\x6f\xd1\x52\x4d\x54\x58\x73\x0f\xf3\x24";

    HMAC_CTX_init(&c);
    HMAC_Init_ex(&c, hmackey, hmackey_size, EVP_sha1(), NULL);
    HMAC_Update(&c, buf, sizeof(buf));
    HMAC_Final(&c, hmac, &hmaclen);
    HMAC_CTX_cleanup(&c);

    if (hmaclen != 20) {
	printf("hmaclen = %d\n", (int)hmaclen);
	return 1;
    }

    if (ct_memcmp(hmac, answer, hmaclen) != 0) {
	printf("wrong answer\n");
	return 1;
    }

    return 0;
}
