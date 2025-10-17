/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 2, 2023.
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
//  gk_reset_check
//
//  Created by Greg on 12/19/14.
//
//

#include <stdio.h>
#include <CoreFoundation/CoreFoundation.h>

int main(int argc, const char * argv[]) {
	// Do not override configuration profiles on users machine
	if (CFPreferencesAppValueIsForced(CFSTR("EnableAssessment"), CFSTR("com.apple.systempolicy.control")) == true ||
		CFPreferencesAppValueIsForced(CFSTR("AllowIdentifiedDevelopers"), CFSTR("com.apple.systempolicy.control")) == true) {
		return 1;
	}
    return 0;
}
