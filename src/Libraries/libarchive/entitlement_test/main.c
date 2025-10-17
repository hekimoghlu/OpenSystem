/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 20, 2023.
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
//
//
//  Created by Justin Vreeland on 6/13/23.
//  Copyright Â© 2023 Apple Inc. All rights reserved.
//

#include <assert.h>
#include <stdlib.h>

#include "archive_check_entitlement.h"


int main(int argc, const char * argv[]) {
	assert(archive_allow_entitlement_format("zip"));
	assert(!archive_allow_entitlement_format("xar"));

	assert(archive_allow_entitlement_filter("compress"));
	assert(!archive_allow_entitlement_filter("bzip2"));

	exit(0);
}
