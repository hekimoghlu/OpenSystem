/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 18, 2023.
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
#include <libkern/c++/OSObject.h>
#include <IOKit/IOReturn.h>

class IOPMPowerSource;

class IOPMPowerSourceList : public OSObject
{
	OSDeclareDefaultStructors(IOPMPowerSourceList);
private:
// pointer to first power source in list
	IOPMPowerSource         *firstItem;

// how many power sources are in the list
	unsigned long           length;

public:
	void initialize(void);
	void free(void) APPLE_KEXT_OVERRIDE;

	unsigned long numberOfItems(void);
	IOReturn addToList(IOPMPowerSource *newPowerSource);
	IOReturn removeFromList(IOPMPowerSource *theItem);

	LIBKERN_RETURNS_NOT_RETAINED IOPMPowerSource *firstInList(void);
	LIBKERN_RETURNS_NOT_RETAINED IOPMPowerSource *nextInList(IOPMPowerSource *currentItem);
};
