/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 12, 2022.
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
#ifndef _IOKIT_IONVRAMCONTROLLER_H
#define _IOKIT_IONVRAMCONTROLLER_H

#include <IOKit/IOService.h>

class IONVRAMController : public IOService
{
	OSDeclareAbstractStructors(IONVRAMController);

public:
	virtual void registerService(IOOptionBits options = 0) APPLE_KEXT_OVERRIDE;

	virtual void sync(void);
	virtual IOReturn select(uint32_t bank);
	virtual IOReturn eraseBank(void);

	virtual IOReturn read(IOByteCount offset, UInt8 *buffer,
	    IOByteCount length) = 0;
	virtual IOReturn write(IOByteCount offset, UInt8 *buffer,
	    IOByteCount length) = 0;
};

#endif /* !_IOKIT_IONVRAMCONTROLLER_H */
