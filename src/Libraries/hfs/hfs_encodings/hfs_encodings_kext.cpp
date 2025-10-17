/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 4, 2022.
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
#include <IOKit/IOService.h>
#include <IOKit/IOLib.h>
#include <sys/queue.h>
#include <libkern/OSKextLib.h>

#include "hfs_encodings_internal.h"

class com_apple_filesystems_hfs_encodings : public IOService {
    OSDeclareDefaultStructors(com_apple_filesystems_hfs_encodings)

public:

	bool start(IOService *provider) override;
	void stop(IOService *provider) override;
};

#define super IOService
OSDefineMetaClassAndStructors(com_apple_filesystems_hfs_encodings, IOService)

bool com_apple_filesystems_hfs_encodings::start(IOService *provider)
{
	if (!super::start(provider))
		return false;

	hfs_converterinit();

	return true;
}

void com_apple_filesystems_hfs_encodings::stop(IOService * provider)
{
	hfs_converterdone();
	super::stop(provider);
}
