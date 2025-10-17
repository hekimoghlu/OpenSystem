/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 1, 2024.
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
//  main.c
//  secdtest
//
//  Created by Fabrice Gautier on 5/29/13.
//
//

#include <stdio.h>
#include <regressions/test/testenv.h>

#include "testlist.h"
#include <regressions/test/testlist_begin.h>
#include "testlist.h"
#include <regressions/test/testlist_end.h>

#include "keychain/ckks/CKKS.h"

#include "featureflags/affordance_featureflags.h"

#include "keychain/securityd/spi.h"

int main(int argc, char * const *argv)
{
    // secdtests should not run any CKKS. It's not entitled for CloudKit, and CKKS threading interferes with many of the tests.
    SecCKKSDisable();

    KCSharingSetChangeTrackingEnabled(false);

    securityd_init(NULL);

    int result = tests_begin(argc, argv);

    fflush(stdout);
    fflush(stderr);

    return result;
}
