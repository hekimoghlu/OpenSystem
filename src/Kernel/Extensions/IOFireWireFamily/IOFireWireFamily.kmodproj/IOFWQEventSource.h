/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 29, 2025.
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
*/

#ifndef _IOKIT_IOFWQEVENTSOURCE_H
#define _IOKIT_IOFWQEVENTSOURCE_H

#import <IOKit/IOEventSource.h>


struct IOFWCmdQ ;
class IOFireWireController ;

class IOFWQEventSource : public IOEventSource
{
    OSDeclareDefaultStructors(IOFWQEventSource)

protected:
    IOFWCmdQ *fQueue;
    virtual bool checkForWork(void) APPLE_KEXT_OVERRIDE;

public:
    bool init(IOFireWireController *owner);
    inline void signalWorkAvailable()	{IOEventSource::signalWorkAvailable();};
    inline void openGate()		{IOEventSource::openGate();};
    inline void closeGate()		{IOEventSource::closeGate();};
	inline bool inGate( void )  {return workLoop->inGate();};
};

#endif
