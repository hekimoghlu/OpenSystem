/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 15, 2022.
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

#include "IOHIDWorkLoop.h"

// system
#include <IOKit/IOWorkLoop.h>
#include <IOKit/IOLocksPrivate.h>

#define super IOWorkLoop
OSDefineMetaClassAndStructors( IOHIDWorkLoop, IOWorkLoop )

static SInt32 gCount = 0;

IOHIDWorkLoop * IOHIDWorkLoop::workLoop()
{
    IOHIDWorkLoop *loop;
    
    loop = new IOHIDWorkLoop;
    if(!loop)
        return loop;
    if(!loop->init()) {
        loop->release();
        loop = NULL;
    }
    return loop;
}

bool
IOHIDWorkLoop::init ( void )
{
	
	SInt32	count = OSIncrementAtomic ( &gCount );
	char	name[64];
	
	snprintf ( name, 64, "HID %d", ( int ) count );
	fLockGroup = lck_grp_alloc_init ( name, LCK_GRP_ATTR_NULL );
	if ( fLockGroup )
	{
		gateLock = IORecursiveLockAllocWithLockGroup ( fLockGroup );
	}
	
	return super::init ( );
	
}

void IOHIDWorkLoop::free ( void )
{
	
	if ( fLockGroup )
	{
		lck_grp_free ( fLockGroup );
		fLockGroup = NULL;
	}
	
	super::free ( );
	
}

