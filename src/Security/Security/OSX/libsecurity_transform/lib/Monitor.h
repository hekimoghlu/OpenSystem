/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 13, 2025.
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

#ifndef __MONITOR__
#define __MONITOR__




#include "Transform.h"



typedef CFTypeRef SecMonitorRef;


class Monitor : public Transform
{
public:
	virtual ~Monitor() { }
	Monitor(CFStringRef mName) : Transform(mName) {}
	virtual void Wait();
	bool IsExternalizable();
};



class BlockMonitor : public Monitor
{
protected:
	dispatch_queue_t mDispatchQueue;
	SecMessageBlock mBlock;
	bool mSeenFinal;

	virtual void AttributeChanged(CFStringRef name, CFTypeRef value);
    void LastValueSent();

	BlockMonitor(dispatch_queue_t queue, SecMessageBlock block);
	
public:
	virtual ~BlockMonitor();
	static CFTypeRef Make(dispatch_queue_t dispatch_queue, SecMessageBlock block);
};


#endif
