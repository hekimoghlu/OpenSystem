/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 17, 2022.
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

// These options must match customrr.m
// TEST_CONFIG MEM=mrc
/*
TEST_BUILD
    $C{COMPILE} $DIR/customrr.m -fvisibility=default -o customrr2.exe -DTEST_EXCHANGEIMPLEMENTATIONS=1 -fno-objc-convert-messages-to-runtime-calls
    $C{COMPILE} -bundle -bundle_loader customrr2.exe $DIR/customrr-cat1.m -o customrr-cat1.bundle
    $C{COMPILE} -bundle -bundle_loader customrr2.exe $DIR/customrr-cat2.m -o customrr-cat2.bundle
END
*/
