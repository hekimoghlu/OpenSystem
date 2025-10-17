/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 26, 2023.
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

#include <pkcs12.h>
#include <evp.h>

struct tests {
    int id;
    const char *password;
    void *salt;
    size_t saltsize;
    int iterations;
    size_t keylen;
    const EVP_MD * (*md)(void);
    void *key;
};

struct tests p12_pbe_tests[] = {
    { PKCS12_KEY_ID,
      NULL,
      "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
      16,
      100,
      16,
      EVP_sha1,
      "\xd7\x2d\xd4\xcf\x7e\xe1\x89\xc5\xb5\xe5\x31\xa7\x63\x2c\xf0\x4b"
    },
    { PKCS12_KEY_ID,
      "",
      "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
      16,
      100,
      16,
      EVP_sha1,
      "\x00\x54\x91\xaf\xc0\x6a\x76\xc3\xf9\xb6\xf2\x28\x1a\x15\xd9\xfe"
    },
    { PKCS12_KEY_ID,
      "foobar",
      "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
      16,
      100,
      16,
      EVP_sha1,
      "\x79\x95\xbf\x3f\x1c\x6d\xe\xe8\xd3\x71\xc4\x94\xd\xb\x18\xb5"
    },
    { PKCS12_KEY_ID,
      "foobar",
      "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
      16,
      2048,
      24,
      EVP_sha1,
      "\x0b\xb5\xe\xa6\x71\x0d\x0c\xf7\x44\xe\xe1\x9b\xb5\xdf\xf1\xdc\x4f\xb0\xca\xe\xee\x4f\xb9\xfd"
    },
    { PKCS12_IV_ID,
      "foobar",
      "\x3c\xdf\x84\x32\x59\xd3\xda\x69",
      8,
      2048,
      8,
      EVP_sha1,
      "\xbf\x9a\x12\xb7\x26\x69\xfd\x05"
    }

};

static int
test_pkcs12_pbe(struct tests *t)
{
    void *key;
    size_t pwlen = 0;

    key = malloc(t->keylen);
    if (t->password)
	pwlen = strlen(t->password);

    if (!PKCS12_key_gen(t->password, pwlen,
			t->salt, t->saltsize,
			t->id, t->iterations, t->keylen,
			key, t->md()))
    {
	printf("key_gen failed\n");
	return 1;
    }

    if (memcmp(t->key, key, t->keylen) != 0) {
	printf("incorrect key\n");
	free(key);
	return 1;
    }
    free(key);
    return 0;
}

int
main(int argc, char **argv)
{
    int ret = 0;
    int i;

    for (i = 0; i < sizeof(p12_pbe_tests)/sizeof(p12_pbe_tests[0]); i++)
	ret += test_pkcs12_pbe(&p12_pbe_tests[i]);

    return ret;
}
