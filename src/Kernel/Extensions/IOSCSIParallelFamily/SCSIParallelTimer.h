/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 31, 2024.
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
#ifndef __IOKIT_SCSI_PARALLEL_TIMER_H__
#define __IOKIT_SCSI_PARALLEL_TIMER_H__


//-----------------------------------------------------------------------------
//	Includes
//-----------------------------------------------------------------------------

// General IOKit includes
#include <IOKit/IOWorkLoop.h>
#include <IOKit/IOTimerEventSource.h>

// SCSI Parallel Family includes
#include "SCSIParallelTask.h"


//-----------------------------------------------------------------------------
//	Constants
//-----------------------------------------------------------------------------

#define kTimeoutValueNone	0


//-----------------------------------------------------------------------------
//	Class Declarations
//-----------------------------------------------------------------------------

class SCSIParallelTimer : public IOTimerEventSource
{
	
	OSDeclareDefaultStructors ( SCSIParallelTimer )
	
public:
	
	static SCSIParallelTimer *
	CreateTimerEventSource ( OSObject * owner, Action action = 0 );
	
	bool				Init ( OSObject * owner, Action action );
	
	void 				Enable ( void );
	void 				Disable ( void );
	void 				CancelTimeout ( void );
	bool				Rearm ( void );
	void				BeginTimeoutContext ( void );
	void				EndTimeoutContext ( void );
	
	IOReturn 			SetTimeout (
							SCSIParallelTaskIdentifier	taskIdentifier,
						  	UInt32					 	timeoutInMS = kTimeoutValueNone );
	
	void				RemoveTask (
							SCSIParallelTaskIdentifier	taskIdentifier );
	
	SCSIParallelTaskIdentifier	GetExpiredTask ( void );
	
	
protected:
	
	inline static SInt32	CompareDeadlines ( AbsoluteTime time1, AbsoluteTime time2 );
	AbsoluteTime			GetDeadline ( SCSIParallelTask * task );
	UInt32					GetTimeoutDuration ( SCSIParallelTask * task );
	
	
private:
	
	queue_head_t			fListHead;
	bool					fHandlingTimeout;
	
};


#endif	/* __IOKIT_SCSI_PARALLEL_TIMER_H__ */