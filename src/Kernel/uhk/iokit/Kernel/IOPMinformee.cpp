/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 10, 2023.
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
#include <IOKit/pwr_mgt/IOPMinformee.h>

#define super OSObject

OSDefineMetaClassAndStructors(IOPMinformee, OSObject)


//*********************************************************************************
// static constructor
//
//*********************************************************************************
IOPMinformee *IOPMinformee::withObject( IOService * theObject )
{
	IOPMinformee        *newInformee = new IOPMinformee;

	if (!newInformee) {
		return NULL;
	}
	newInformee->init();
	newInformee->initialize( theObject );

	return newInformee;
}


//*********************************************************************************
// constructor
//
//*********************************************************************************
void
IOPMinformee::initialize( IOService * theObject )
{
	whatObject = theObject;
	timer = 0;
	active = true;
	whatObject->retain();
}


//*********************************************************************************
// free
//
//*********************************************************************************
void
IOPMinformee::free(void )
{
	whatObject->release();
	super::free();
}
