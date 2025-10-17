/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 2, 2023.
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
#ifndef _IOKIT_IOFWASYNCSTREAMRECEIVEPORT_H
#define _IOKIT_IOFWASYNCSTREAMRECEIVEPORT_H

#include <IOKit/IOService.h>
#include <IOKit/firewire/IOFWLocalIsochPort.h>

class IOFireWireController;

class IOFWAsyncStreamReceivePort : public IOFWLocalIsochPort
{
    OSDeclareDefaultStructors(IOFWAsyncStreamReceivePort)

private:
	
	UInt32	fChannel ;
		
public:
    virtual bool init(IODCLProgram *program, IOFireWireController *control, UInt32 channel);

	// Return maximum speed and channels supported
	// (bit n set = chan n supported)
    IOReturn getSupported(IOFWSpeed &maxSpeed, UInt64 &chanSupported) APPLE_KEXT_OVERRIDE;
};

#endif /* _IOKIT_IOFWASYNCSTREAMRECEIVEPORT_H */

