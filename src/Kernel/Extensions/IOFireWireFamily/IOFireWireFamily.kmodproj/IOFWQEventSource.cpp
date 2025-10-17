/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 12, 2024.
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
	$Log: not supported by cvs2svn $
	Revision 1.3  2002/10/18 23:29:42  collin
	fix includes, fix cast which fails on new compiler
	
	Revision 1.2  2002/09/25 00:27:20  niels
	flip your world upside-down
	
*/

// public
#import <IOKit/firewire/IOFireWireController.h>

// private
#import "IOFWQEventSource.h"

// IOFWQEventSource
OSDefineMetaClassAndStructors(IOFWQEventSource, IOEventSource)

// checkForWork
//
//

bool IOFWQEventSource::checkForWork()
{
    return fQueue->executeQueue(false);
}

// init
//
//

bool IOFWQEventSource::init( IOFireWireController* owner )
{
    fQueue = &owner->getPendingQ();
    return IOEventSource::init(owner);
}
