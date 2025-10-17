/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 27, 2022.
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
#ifndef _IOKIT_IOPMINFORMEE_H
#define _IOKIT_IOPMINFORMEE_H

#include <IOKit/IOService.h>
#include <IOKit/IOReturn.h>

class IOPMinformee : public OSObject
{
	OSDeclareDefaultStructors(IOPMinformee);
	friend class IOPMinformeeList;

public:
	static IOPMinformee * withObject( IOService * theObject );

	void initialize( IOService * theObject );

	void free( void ) APPLE_KEXT_OVERRIDE;

public:
	IOService *     whatObject; // interested driver
	int32_t         timer;      // -1, 0, or positive number of ticks
	IOPMinformee *  nextInList; // linkage pointer
	AbsoluteTime    startTime;  // start time of last inform
	bool            active;     // enable flag
};

#endif /* !_IOKIT_IOPMINFORMEE_H */
