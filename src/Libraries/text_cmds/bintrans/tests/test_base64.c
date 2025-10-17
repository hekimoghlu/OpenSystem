/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 14, 2021.
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

//
//  uuencode.c
//  basic_cmds
//
//  base64 tests
//
//

#include <stdint.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/types.h>

#include <darwintest.h>
#include <darwintest_utils.h>

int apple_b64_ntop(u_char const *, size_t, char *, size_t);

T_DECL(b64_ntop_basic,
       "Test basic functionality of b64_ntop and measure the performance",
       T_META_CHECK_LEAKS(NO))
{
    char b64_output[512] = {0};
    uint8_t bin_input[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 'a', 'z', 'f' ,'o', 'o', '#'};

    int retval = apple_b64_ntop((const u_char *)bin_input, 16, b64_output, 512);
    T_ASSERT_GT(retval, 0, "Successful conversion");

    T_LOG("Output: %s", b64_output);
    T_ASSERT_EQ_STR(b64_output, "AQIDBAUGBwgJAGF6Zm9vIw==", "Encoding operation of 16 bytes");

    dt_stat_time_t s = dt_stat_time_create("apple_b64_ntop");

    T_STAT_MEASURE_LOOP(s) {
        apple_b64_ntop((const u_char *)bin_input, 16, b64_output, 512);
    }

    dt_stat_finalize(s);
}

int apple_b64_pton(char const *, u_char *, size_t);

T_DECL(apple_b64_pton_basic,
       "Test basic functionality of apple_b64_pton and measure the performance",
       T_META_CHECK_LEAKS(NO))
{
    const char *b64_input = "AQIDBAUGBwgJAGF6Zm9vIw==";
    uint8_t bin_output[64] = {0};
    const uint8_t bin_expected[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 'a', 'z', 'f', 'o', 'o', '#'};

    int retval = apple_b64_pton((const char *)b64_input, bin_output, 64);
    T_ASSERT_GT(retval, 0, "Successful conversion");

    int i;
    for (i = 0; i < 64; ++i) {
        if (i < 16) {
            if (bin_output[i] != bin_expected[i]) {
                T_LOG("bin_output[%d] != bin_expected[%d] (%d != %d)",
                      i, i, bin_output[i], bin_expected[i]);
                break;
            }
        } else if (bin_output[i]) {
            T_LOG("bin_output[%d] = %d", i, bin_output[i]);
            break;
        }
    }
    T_ASSERT_EQ_INT(64, i, "The converted data is as expected");

    dt_stat_time_t s = dt_stat_time_create("apple_b64_pton");

    T_STAT_MEASURE_LOOP(s) {
        apple_b64_pton((const char *)b64_input, bin_output, 64);
    }

    dt_stat_finalize(s);
}
