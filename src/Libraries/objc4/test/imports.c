/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 11, 2024.
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
/*
TEST_BUILD
echo $C{XCRUN} nm -m -arch $C{ARCH} $C{TESTLIB}
test -e $C{TESTLIB} && $C{XCRUN} nm -u -m -arch $C{ARCH} $C{TESTLIB} | grep -v 'weak external ____chkstk_darwin \(from libSystem\)' | grep -v 'weak external _objc_bp_assist_cfg_np \(from libSystem\)' | grep -v 'weak external __objc_test_get_environ \(from libobjc-env\)' | egrep '(weak external| external (___cxa_atexit|___cxa_guard_acquire|___cxa_guard_release))' || true
test -e $C{TESTLIB} && $C{XCRUN} nm -U -m -arch $C{ARCH} $C{TESTLIB} | egrep '(weak external| external __Z)' || true
$C{COMPILE_C} $DIR/imports.c -o imports.exe
END

TEST_BUILD_OUTPUT
.*libobjc.A.dylib
END
 */

#include "test.h"
int main()
{
    succeed(__FILE__);
}
