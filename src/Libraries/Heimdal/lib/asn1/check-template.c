/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 30, 2022.
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

#include <stdio.h>
#include <string.h>
#include <err.h>
#include <roken.h>

#include <asn1-common.h>
#include <asn1_err.h>
#include <der.h>
#include <test_asn1.h>

#include "check-common.h"

static int
cmp_dummy (void *a, void *b)
{
    return 0;
}

static int
test_seqofseq(void)
{
    struct test_case tests[] = {
	{ NULL,  2,
	  "\x30\x00",
	  "seqofseq 0" },
	{ NULL,  9,
	  "\x30\x07\x30\x05\xa0\x03\x02\x01\x00",
	  "seqofseq 1" },
	{ NULL,  16,
	  "\x30\x0e\x30\x05\xa0\x03\x02\x01\x00\x30\x05\xa0\x03\x02\x01\x01",
	  "seqofseq 2" }
    };

    int ret = 0, ntests = sizeof(tests) / sizeof(*tests);
    TESTSeqOfSeq c0, c1, c2;
    struct TESTSeqOfSeq_val i[2];

    i[0].zero = 0;
    i[1].zero = 1;

    c0.len = 0;
    c0.val = NULL;
    tests[0].val = &c0;

    c1.len = 1;
    c1.val = i;
    tests[1].val = &c1;

    c2.len = 2;
    c2.val = i;
    tests[2].val = &c2;

    ret += generic_test (tests, ntests, sizeof(TESTSeqOfSeq),
			 (generic_encode)encode_TESTSeqOfSeq,
			 (generic_length)length_TESTSeqOfSeq,
			 (generic_decode)decode_TESTSeqOfSeq,
			 (generic_free)free_TESTSeqOfSeq,
			 cmp_dummy,
			 NULL);
    return ret;
}

static int
test_seqofseq2(void)
{
    struct test_case tests[] = {
	{ NULL,  2,
	  "\x30\x00",
	  "seqofseq2 0" },
	{ NULL,  11,
	  "\x30\x09\x30\x07\xa0\x05\x1b\x03\x65\x74\x74",
	  "seqofseq2 1" },
	{ NULL,  21,
	  "\x30\x13\x30\x07\xa0\x05\x1b\x03\x65\x74\x74\x30\x08\xa0"
	  "\x06\x1b\x04\x74\x76\x61\x61",
	  "seqofseq2 2" }
    };

    int ret = 0, ntests = sizeof(tests) / sizeof(*tests);
    TESTSeqOfSeq2 c0, c1, c2;
    struct TESTSeqOfSeq2_val i[2];

    i[0].string = "ett";
    i[1].string = "tvaa";

    c0.len = 0;
    c0.val = NULL;
    tests[0].val = &c0;

    c1.len = 1;
    c1.val = i;
    tests[1].val = &c1;

    c2.len = 2;
    c2.val = i;
    tests[2].val = &c2;

    ret += generic_test (tests, ntests, sizeof(TESTSeqOfSeq2),
			 (generic_encode)encode_TESTSeqOfSeq2,
			 (generic_length)length_TESTSeqOfSeq2,
			 (generic_decode)decode_TESTSeqOfSeq2,
			 (generic_free)free_TESTSeqOfSeq2,
			 cmp_dummy,
			 NULL);
    return ret;
}

static int
test_seqof2(void)
{
    struct test_case tests[] = {
	{ NULL,  4,
	  "\x30\x02\x30\x00",
	  "seqof2 1" },
	{ NULL,  9,
	  "\x30\x07\x30\x05\x1b\x03\x66\x6f\x6f",
	  "seqof2 2" },
	{ NULL,  14,
	  "\x30\x0c\x30\x0a\x1b\x03\x66\x6f\x6f\x1b\x03\x62\x61\x72",
	  "seqof2 3" }
    };

    int ret = 0, ntests = sizeof(tests) / sizeof(*tests);
    TESTSeqOf2 c0, c1, c2;
    heim_general_string i[2];

    i[0] = "foo";
    i[1] = "bar";

    c0.strings.val = NULL;
    c0.strings.len = 0;
    tests[0].val = &c0;

    c1.strings.len = 1;
    c1.strings.val = i;
    tests[1].val = &c1;

    c2.strings.len = 2;
    c2.strings.val = i;
    tests[2].val = &c2;

    ret += generic_test (tests, ntests, sizeof(TESTSeqOf2),
			 (generic_encode)encode_TESTSeqOf2,
			 (generic_length)length_TESTSeqOf2,
			 (generic_decode)decode_TESTSeqOf2,
			 (generic_free)free_TESTSeqOf2,
			 cmp_dummy,
			 NULL);
    return ret;
}

static int
test_seqof3(void)
{
    struct test_case tests[] = {
	{ NULL,  2,
	  "\x30\x00",
	  "seqof3 0" },
	{ NULL,  4,
	  "\x30\x02\x30\x00",
	  "seqof3 1" },
	{ NULL,  9,
	  "\x30\x07\x30\x05\x1b\x03\x66\x6f\x6f",
	  "seqof3 2" },
	{ NULL,  14,
	  "\x30\x0c\x30\x0a\x1b\x03\x66\x6f\x6f\x1b\x03\x62\x61\x72",
	  "seqof3 3" }
    };

    int ret = 0, ntests = sizeof(tests) / sizeof(*tests);
    TESTSeqOf3 c0, c1, c2, c3;
    struct TESTSeqOf3_strings s1, s2, s3;
    heim_general_string i[2];

    i[0] = "foo";
    i[1] = "bar";

    c0.strings = NULL;
    tests[0].val = &c0;

    s1.val = NULL;
    s1.len = 0;
    c1.strings = &s1;
    tests[1].val = &c1;

    s2.len = 1;
    s2.val = i;
    c2.strings = &s2;
    tests[2].val = &c2;

    s3.len = 2;
    s3.val = i;
    c3.strings = &s3;
    tests[3].val = &c3;

    ret += generic_test (tests, ntests, sizeof(TESTSeqOf3),
			 (generic_encode)encode_TESTSeqOf3,
			 (generic_length)length_TESTSeqOf3,
			 (generic_decode)decode_TESTSeqOf3,
			 (generic_free)free_TESTSeqOf3,
			 cmp_dummy,
			 NULL);
    return ret;
}


int
main(int argc, char **argv)
{
    int ret = 0;

    ret += test_seqofseq();
    ret += test_seqofseq2();
    ret += test_seqof2();
    ret += test_seqof3();

    return ret;
}
