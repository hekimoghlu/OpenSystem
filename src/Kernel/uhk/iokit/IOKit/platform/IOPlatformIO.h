/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 5, 2022.
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
#ifndef _IOKIT_PLATFORM_IOPLATFORMIO_H
#define _IOKIT_PLATFORM_IOPLATFORMIO_H

extern "C" {
#include <kern/kern_types.h>
}

#include <IOKit/IOService.h>

/*!
 * @class      IOPlatformIO
 * @abstract   The base class for platform I/O drivers, such as AppleARMIO.
 */
class IOPlatformIO : public IOService
{
	OSDeclareAbstractStructors(IOPlatformIO);

public:
	virtual bool start(IOService * provider) APPLE_KEXT_OVERRIDE;

	/*!
	 * @function   handlePlatformError
	 * @abstract   Handler for platform-defined errors.
	 * @discussion If the CPU reports an error that XNU does not know how
	 *             to handle, such as a parity error or SError, XNU will
	 *             invoke this method if there is an IOPlatformIO
	 *             driver loaded.
	 * @param far  Fault address provided by the CPU, if any.
	 * @result     true if the exception was handled, false if not.
	 */
	virtual bool handlePlatformError(vm_offset_t far) = 0;
};

#endif /* ! _IOKIT_PLATFORM_IOPLATFORMIO_H */
