/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 13, 2025.
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
#include <stdlib.h>

#include "test_regressions.h"

int test_00_test(int argc, char *const *argv)
{
    int rv = 1;
    plan_tests(6);

    TODO: {
	todo("ok 0 is supposed to fail");

	rv = ok(0, "ok bad");
	if (!rv)
	    diag("ok bad not good today");
    }
    rv &= ok(1, "ok ok");
    (void) rv;
#if 0
    SKIP: {
	skip("is bad will fail", 1, 0);

	if (!is(0, 4, "is bad"))
	    diag("is bad not good today");
    }
    SKIP: {
	skip("is ok should not be skipped", 1, 1);

        is(3, 3, "is ok");
    }
#endif
    isnt(0, 4, "isnt ok");
    TODO: {
	todo("isnt bad is supposed to fail");

	isnt(3, 3, "isnt bad");
    }
    TODO: {
	todo("cmp_ok bad is supposed to fail");

	cmp_ok(3, &&, 0, "cmp_ok bad");
    }
    cmp_ok(3, &&, 3, "cmp_ok ok");

    return 0;
}
