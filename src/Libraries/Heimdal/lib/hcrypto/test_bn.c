/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 6, 2024.
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

#include <bn.h>
#include <rand.h>

static int
set_get(unsigned long num)
{
    BIGNUM *bn;

    bn = BN_new();
    if (!BN_set_word(bn, num))
	return 1;

    if (BN_get_word(bn) != num)
	return 1;

    BN_free(bn);
    return 0;
}

#define CHECK(x) do { ret += x; } while(0)

static int
test_BN_set_get(void)
{
    int ret = 0;
    CHECK(set_get(0));
    CHECK(set_get(1));
    CHECK(set_get(0xff));
    CHECK(set_get(0x1ff));
    CHECK(set_get(0xffff));
    CHECK(set_get(0xf000));
    CHECK(set_get(ULONG_MAX / 2));
    CHECK(set_get(ULONG_MAX - 1));

    return ret;
}

static int
test_BN_bit(void)
{
    BIGNUM *bn;
    int ret = 0;

    bn = BN_new();

    /* test setting and getting of "word" */
    if (!BN_set_word(bn, 1))
	return 1;
    if (!BN_is_bit_set(bn, 0))
	ret += 1;
    if (!BN_is_bit_set(bn, 0))
	ret += 1;

    if (!BN_set_word(bn, 2))
	return 1;
    if (!BN_is_bit_set(bn, 1))
	ret += 1;

    if (!BN_set_word(bn, 3))
	return 1;
    if (!BN_is_bit_set(bn, 0))
	ret += 1;
    if (!BN_is_bit_set(bn, 1))
	ret += 1;

    if (!BN_set_word(bn, 0x100))
	return 1;
    if (!BN_is_bit_set(bn, 8))
	ret += 1;

    if (!BN_set_word(bn, 0x1000))
	return 1;
    if (!BN_is_bit_set(bn, 12))
	ret += 1;

    /* test bitsetting */
    if (!BN_set_word(bn, 1))
	return 1;
    if (!BN_set_bit(bn, 1))
	return 1;
    if (BN_get_word(bn) != 3)
	return 1;
    if (!BN_clear_bit(bn, 0))
	return 1;
    if (BN_get_word(bn) != 2)
	return 1;

    /* test bitsetting past end of current end */
    BN_clear(bn);
    if (!BN_set_bit(bn, 12))
	return 1;
    if (BN_get_word(bn) != 0x1000)
	return 1;

    /* test bit and byte counting functions */
    if (BN_num_bits(bn) != 13)
	return 1;
    if (BN_num_bytes(bn) != 2)
	return 1;

    BN_free(bn);
    return ret;
}

struct ietest {
    char *data;
    size_t len;
    unsigned long num;
} ietests[] = {
    { "", 0, 0 },
    { "\x01", 1, 1 },
    { "\x02", 1, 2 },
    { "\xf2", 1, 0xf2 },
    { "\x01\x00", 2, 256 }
};

static int
test_BN_import_export(void)
{
    BIGNUM *bn;
    int ret = 0;
    int i;

    bn = BN_new();

    for (i = 0; i < sizeof(ietests)/sizeof(ietests[0]); i++) {
	size_t len;
	unsigned char *p;
	if (!BN_bin2bn((unsigned char*)ietests[i].data, ietests[i].len, bn))
	    return 1;
	if (BN_get_word(bn) != ietests[i].num)
	    return 1;
	len = BN_num_bytes(bn);
	if (len != ietests[i].len)
	    return 1;
	p = malloc(len + 1);
	p[len] = 0xf4;
	BN_bn2bin(bn, p);
	if (p[len] != 0xf4)
	    return 1;
	if (memcmp(p, ietests[i].data, ietests[i].len) != 0)
	    return 1;
	free(p);
    }

    BN_free(bn);
    return ret;
}

static int
test_BN_uadd(void)
{
    BIGNUM *a, *b, *c;
    char *p;

    a = BN_new();
    b = BN_new();
    c = BN_new();

    BN_set_word(a, 1);
    BN_set_word(b, 2);

    BN_uadd(c, a, b);

    if (BN_get_word(c) != 3)
	return 1;

    BN_uadd(c, b, a);

    if (BN_get_word(c) != 3)
	return 1;

    BN_set_word(b, 0xff);

    BN_uadd(c, a, b);
    if (BN_get_word(c) != 0x100)
	return 1;

    BN_uadd(c, b, a);
    if (BN_get_word(c) != 0x100)
	return 1;

    BN_set_word(a, 0xff);

    BN_uadd(c, a, b);
    if (BN_get_word(c) != 0x1fe)
	return 1;

    BN_uadd(c, b, a);
    if (BN_get_word(c) != 0x1fe)
	return 1;


    BN_free(a);
    BN_free(b);

    BN_hex2bn(&a, "50212A3B611D46642C825A16A354CE0FD4D85DD2");
    BN_hex2bn(&b, "84B6C7E8D28ACA1614954DA");

    BN_uadd(c, b, a);
    p = BN_bn2hex(c);
    if (strcmp(p, "50212A3B611D466434CDC695307D7AB13621B2AC") != 0) {
	free(p);
	return 1;
    }
    free(p);

    BN_uadd(c, a, b);
    p = BN_bn2hex(c);
    if (strcmp(p, "50212A3B611D466434CDC695307D7AB13621B2AC") != 0) {
	free(p);
	return 1;
    }
    free(p);

    BN_free(a);
    BN_free(b);
    BN_free(c);

    return 0;
}

static int
test_BN_cmp(void)
{
    BIGNUM *a, *b;

    a = BN_new();
    b = BN_new();

    if (!BN_set_word(a, 1))
	return 1;
    if (!BN_set_word(b, 1))
	return 1;

    if (BN_cmp(a, b) != 0)
	return 1;
    if (BN_cmp(b, a) != 0)
	return 1;

    if (!BN_set_word(b, 2))
	return 1;

    if (BN_cmp(a, b) >= 0)
	return 1;
    if (BN_cmp(b, a) <= 0)
	return 1;

    BN_set_negative(b, 1);

    if (BN_cmp(a, b) <= 0)
	return 1;
    if (BN_cmp(b, a) >= 0)
	return 1;

    BN_free(a);
    BN_free(b);

    BN_hex2bn(&a, "50212A3B611D46642C825A16A354CE0FD4D85DD1");
    BN_hex2bn(&b, "50212A3B611D46642C825A16A354CE0FD4D85DD2");

    if (BN_cmp(a, b) >= 0)
	return 1;
    if (BN_cmp(b, a) <= 0)
	return 1;

    BN_set_negative(b, 1);

    if (BN_cmp(a, b) <= 0)
	return 1;
    if (BN_cmp(b, a) >= 0)
	return 1;

    BN_free(a);
    BN_free(b);
    return 0;
}

static int
test_BN_rand(void)
{
    BIGNUM *bn;

    if (RAND_status() != 1)
	return 0;

    bn = BN_new();
    if (bn == NULL)
	return 1;

    if (!BN_rand(bn, 1024, 0, 0))
	return 1;

    BN_free(bn);
    return 0;
}

#define testnum 100
#define testnum2 10

static int
test_BN_CTX(void)
{
    unsigned int i, j;
    BIGNUM *bn;
    BN_CTX *c;

    if ((c = BN_CTX_new()) == NULL)
	return 1;

    for (i = 0; i < testnum; i++) {
	BN_CTX_start(c);
	BN_CTX_end(c);
    }

    for (i = 0; i < testnum; i++)
	BN_CTX_start(c);
    for (i = 0; i < testnum; i++)
	BN_CTX_end(c);

    for (i = 0; i < testnum; i++) {
	BN_CTX_start(c);
	if ((bn = BN_CTX_get(c)) == NULL)
	    return 1;
	BN_CTX_end(c);
    }

    for (i = 0; i < testnum; i++) {
	BN_CTX_start(c);
	for (j = 0; j < testnum2; j++)
	    if ((bn = BN_CTX_get(c)) == NULL)
		return 1;
    }
    for (i = 0; i < testnum; i++)
	BN_CTX_end(c);

    BN_CTX_free(c);
    return 0;
}


int
main(int argc, char **argv)
{
    int ret = 0;

    ret += test_BN_set_get();
    ret += test_BN_bit();
    ret += test_BN_import_export();
    ret += test_BN_uadd();
    ret += test_BN_cmp();
    ret += test_BN_rand();
    ret += test_BN_CTX();

    return ret;
}
