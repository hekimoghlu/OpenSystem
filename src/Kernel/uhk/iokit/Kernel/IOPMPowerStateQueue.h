/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 20, 2022.
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
#ifndef _IOPMPOWERSTATEQUEUE_H_
#define _IOPMPOWERSTATEQUEUE_H_

#include <IOKit/IOEventSource.h>
#include <IOKit/IOLocks.h>
#include <kern/queue.h>

typedef void (*IOPMPowerStateQueueAction)(OSObject *, uint32_t event, void *, uint64_t);

class IOPMPowerStateQueue : public IOEventSource
{
	OSDeclareDefaultStructors(IOPMPowerStateQueue);

private:
	struct PowerEventEntry {
		queue_chain_t   chain;
		uint32_t        eventType;
		void *          arg0;
		uint64_t        arg1;
	};

	queue_head_t    queueHead;
	IOLock *        queueLock;

protected:
	virtual bool checkForWork( void ) APPLE_KEXT_OVERRIDE;
	virtual bool init( OSObject * owner, Action action ) APPLE_KEXT_OVERRIDE;

public:
	static IOPMPowerStateQueue * PMPowerStateQueue( OSObject * owner, Action action );

	bool submitPowerEvent( uint32_t eventType, void * arg0 = NULL, uint64_t arg1 = 0 );
};

#endif /* _IOPMPOWERSTATEQUEUE_H_ */
