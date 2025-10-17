/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 4, 2022.
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

#ifndef __SINGLE_SHOT_SOURCE__
#define __SINGLE_SHOT_SOURCE__

#include "Source.h"

extern CFStringRef gSingleShotSourceName;

/*
	We need this source because we need to send the data followed by
	a null value, so that all input sources have the same behavior.
*/

class SingleShotSource : public Source
{
protected:
	SingleShotSource(CFTypeRef value, Transform* t, CFStringRef name);

public:
	void DoActivate();
	Boolean Equal(const CoreFoundationObject* obj);
	static CFTypeRef Make(CFTypeRef value, Transform* t, CFStringRef name);
	std::string DebugDescription();
};

#endif
