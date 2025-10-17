/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 26, 2023.
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
//  escape2.m
//  btest
//
//  Created by Apple on 6/12/08.
//  Copyright 2008 __MyCompanyName__. All rights reserved.
//


#import "common.h"




void test(void) {
	// validate that escaping a context is enough
	int counter = 0;
	while (counter < 10) {
		BYREF int i = 0;
		vv block = ^{  ++i; };
		if (counter > 5) {
			lastUse(i);
			break;
		}
		++counter;
		lastUse(i);
	}
}
