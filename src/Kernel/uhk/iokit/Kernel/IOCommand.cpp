/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 12, 2023.
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
 *
 *	Copyright (c) 2000 Apple Computer, Inc.  All rights reserved.
 *
 *	HISTORY
 *
 *	11/13/2000		CJS		Created IOCommand class and implementation
 *
 */

#include <IOKit/IOCommand.h>

#define super OSObject
OSDefineMetaClassAndStructors(IOCommand, OSObject);


//--------------------------------------------------------------------------
//	init -	initialize our data structures
//--------------------------------------------------------------------------

bool
IOCommand::init(void)
{
	if (super::init()) {
		queue_init(&fCommandChain);
		return true;
	} else {
		return false;
	}
}
