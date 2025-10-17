/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 22, 2023.
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
 * Copyright (c) 2001 Apple Computer, Inc.  All rights reserved.
 *
 * HISTORY
 *
 */


#ifndef _IOKIT_IOFWWORKLOOP_H
#define _IOKIT_IOFWWORKLOOP_H

#include <IOKit/IOWorkLoop.h>
#include <libkern/c++/OSSet.h>

class IOFWWorkLoop : public IOWorkLoop
{
    OSDeclareDefaultStructors(IOFWWorkLoop)

protected:
    void *				fSleepToken;
	static SInt32		sLockGroupCount;
	lck_grp_t *			fLockGroup;

	IOThread			fRemoveSourceThread;
	OSSet *				fRemoveSourceDeferredSet;
	
	bool init( void ) APPLE_KEXT_OVERRIDE;
	void free( void ) APPLE_KEXT_OVERRIDE;
	
    // Overrides to check for sleeping
    virtual void closeGate() APPLE_KEXT_OVERRIDE;
    virtual bool tryCloseGate() APPLE_KEXT_OVERRIDE;
	
public:
    // Create a workloop
    static IOFWWorkLoop * workLoop();
    
    // Put workloop to sleep (Must have gate closed, opens gate if successful)
    virtual IOReturn sleep( void *token );
    
    // Wake workloop up (closes gate if successful)
    virtual IOReturn wake( void *token );
	
	virtual IOReturn removeEventSource(IOEventSource *toRemove) APPLE_KEXT_OVERRIDE;

};

#endif /* ! _IOKIT_IOFWWORKLOOP_H */

