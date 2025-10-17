/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 7, 2024.
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
#include <sys/types.h>
#include <inttypes.h>
#include <roken.h>
#define HEIM_FUZZER_INTERNALS 1
#include "fuzzer.h"

#define SIZE(_array) (sizeof(_array) / sizeof((_array)[0]))

/*
 *
 */

static void
null_free(void *ctx)
{
    if (ctx != NULL)
	abort();
}

/*
 *
 */

#define MIN_RANDOM_TRIES 30000

static unsigned long
random_tries(size_t length)
{
    length = length * 12;
    if (length < MIN_RANDOM_TRIES)
	length = MIN_RANDOM_TRIES;
    return (unsigned long)length;
}

static int
random_fuzz(void **ctx, unsigned long iteration, uint8_t *data, size_t length)
{
    *ctx = NULL;

    if (iteration > MIN_RANDOM_TRIES && iteration > length * 12)
	return 1;

    data[rk_random() % length] = rk_random();

    return 0;
}


const struct heim_fuzz_type_data __heim_fuzz_random = {
    "random",
    random_tries,
    random_fuzz,
    null_free
};

/*
 *
 */

static unsigned long
bitflip_tries(size_t length)
{
    return length << 3;
}

static int
bitflip_fuzz(void **ctx, unsigned long iteration, uint8_t *data, size_t length)
{
    *ctx = NULL;
    if ((iteration >> 3) >= length)
	return 1;
    data[iteration >> 3] ^= (1 << (iteration & 7));

    return 0;
}

const struct heim_fuzz_type_data __heim_fuzz_bitflip = {
    "bitflip",
    bitflip_tries,
    bitflip_fuzz,
    null_free
};

/*
 *
 */

static unsigned long
byteflip_tries(size_t length)
{
    return length;
}

static int
byteflip_fuzz(void **ctx, unsigned long iteration, uint8_t *data, size_t length)
{
    *ctx = NULL;
    if (iteration >= length)
	return 1;
    data[iteration] ^= 0xff;

    return 0;
}

const struct heim_fuzz_type_data __heim_fuzz_byteflip = {
    "byteflip",
    byteflip_tries,
    byteflip_fuzz,
    null_free
};

/*
 *
 */

static unsigned long
shortflip_tries(size_t length)
{
    return length / 2;
}

static int
shortflip_fuzz(void **ctx, unsigned long iteration, uint8_t *data, size_t length)
{
    *ctx = NULL;
    if (iteration + 1 >= length / 2)
	return 1;
    data[iteration + 0] ^= 0xff;
    data[iteration + 1] ^= 0xff;

    return 0;
}

const struct heim_fuzz_type_data __heim_fuzz_shortflip = {
    "shortflip",
    shortflip_tries,
    shortflip_fuzz,
    null_free
};

/*
 *
 */

static unsigned long
wordflip_tries(size_t length)
{
    return length / 4;
}

static int
wordflip_fuzz(void **ctx, unsigned long iteration, uint8_t *data, size_t length)
{
    if (ctx)
	*ctx = NULL;
    if (iteration + 3 >= length / 4)
	return 1;
    data[iteration + 0] ^= 0xff;
    data[iteration + 1] ^= 0xff;
    data[iteration + 2] ^= 0xff;
    data[iteration + 3] ^= 0xff;

    return 0;
}

const struct heim_fuzz_type_data __heim_fuzz_wordflip = {
    "wordflip",
    wordflip_tries,
    wordflip_fuzz,
    null_free
};

/*
 * interesting values picked from AFL
 */

static uint8_t interesting_u8[] = {
    -128,
    -1,
    0,
    1,
    16,
    32,
    64,
    100,
    127
};

static uint16_t interesting_u16[] = {
    (uint16_t)-32768,
    (uint16_t)-129,
    128,
    255,
    256,
    512,
    1000,
    1024,
    4096,
    32767
};

static uint32_t interesting_u32[] = {
    (uint32_t)-2147483648LL,
    (uint32_t)-100000000,
    (uint32_t)-32769,
    32768,
   65535,
   65536,
   100000000,
   2147483647
};


static unsigned long
interesting_tries(size_t length)
{
    return length;
}

static int
interesting8_fuzz(void **ctx, unsigned long iteration, uint8_t *data, size_t length)
{
    if (length < iteration / SIZE(interesting_u8))
	return 1;

    memcpy(&data[iteration % SIZE(interesting_u8)], &interesting_u8[iteration / SIZE(interesting_u8)], 1);
    return 0;
}

const struct heim_fuzz_type_data __heim_fuzz_interesting8 = {
    "interesting uint8",
    interesting_tries,
    interesting8_fuzz,
    null_free
};

static int
interesting16_fuzz(void **ctx, unsigned long iteration, uint8_t *data, size_t length)
{
    if (length < 1 + (iteration / SIZE(interesting_u16)))
	return 1;

    memcpy(&data[iteration % SIZE(interesting_u16)], &interesting_u16[iteration / SIZE(interesting_u16)], 2);
    return 0;
}

const struct heim_fuzz_type_data __heim_fuzz_interesting16 = {
    "interesting uint16",
    interesting_tries,
    interesting16_fuzz,
    null_free
};

static int
interesting32_fuzz(void **ctx, unsigned long iteration, uint8_t *data, size_t length)
{
    if (length < 3 + (iteration / SIZE(interesting_u32)))
	return 1;

    memcpy(&data[iteration % SIZE(interesting_u32)], &interesting_u32[iteration / SIZE(interesting_u32)], 4);
    return 0;
}

const struct heim_fuzz_type_data __heim_fuzz_interesting32 = {
    "interesting uint32",
    interesting_tries,
    interesting32_fuzz,
    null_free
};

/*
 *
 */

const char *
heim_fuzzer_name(heim_fuzz_type_t type)
{
    return type->name;
}

unsigned long
heim_fuzzer_tries(heim_fuzz_type_t type, size_t length)
{
    return type->tries(length);
}

int
heim_fuzzer(heim_fuzz_type_t type,
	    void **ctx, 
	    unsigned long iteration,
	    uint8_t *data,
	    size_t length)
{
    if (length == 0)
	return 1;
    return type->fuzz(ctx, iteration, data, length);
}

void
heim_fuzzer_free(heim_fuzz_type_t type,
		 void *ctx)
{
    if (ctx != NULL)
	type->freectx(ctx);
}
