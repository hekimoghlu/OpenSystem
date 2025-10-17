/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 1, 2025.
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
#include <der_locl.h>

RCSID("$Id$");

static int
test_timegm(void)
{
    int ret = 0;
    struct tm tm;
    time_t t;

    memset(&tm, 0, sizeof(tm));
    tm.tm_year = 106;
    tm.tm_mon = 9;
    tm.tm_mday = 1;
    tm.tm_hour = 10;
    tm.tm_min = 3;

    t = _der_timegm(&tm);
    if (t != 1159696980)
	ret += 1;

    tm.tm_sec = 60;
    t = _der_timegm(&tm);
    if (t != 1159697040)
        ret += 1;

    tm.tm_mday = 0;
    t = _der_timegm(&tm);
    if (t != -1)
	ret += 1;

    _der_gmtime(1159696980, &tm);
    if (tm.tm_year != 106 ||
	tm.tm_mon != 9 ||
	tm.tm_mday != 1 ||
	tm.tm_hour != 10 ||
	tm.tm_min != 3 ||
	tm.tm_sec != 0)
      errx(1, "tmtime fails");

    return ret;
}

int
main(int argc, char **argv)
{
    int ret = 0;

    ret += test_timegm();

    return ret;
}
