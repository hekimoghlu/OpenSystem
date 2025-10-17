/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 21, 2022.
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

//
// Glue to make IO Kit happy with BootCache as a non-NKE KEXT.
//

#include <IOKit/IOService.h>

/*
 * Hooks from c++ glue to the cache core.
 */
extern "C" void	BC_load(void);
extern "C" int	BC_unload(void);

class com_apple_BootCache : public IOService
{
	OSDeclareDefaultStructors(com_apple_BootCache);

public:
	virtual bool	start(IOService *provider);
	virtual void	stop(IOService *provider);
};

OSDefineMetaClassAndStructors(com_apple_BootCache, IOService);

bool
com_apple_BootCache::start(IOService *provider)
{
	bool	result;

	result = IOService::start(provider);
	if (result == true) {
		BC_load();
#ifndef __clang_analyzer__ // Avoid false positive leak (released in stop function)
		provider->retain();
#endif
	}
	return(result);
}

void
com_apple_BootCache::stop(IOService *provider)
{
	if (BC_unload())
		return;	// refuse unload?
	IOService::stop(provider);
#ifndef __clang_analyzer__ // Avoid false positive overdecrement (retained in start function)
	provider->release();
#endif
}
