/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 27, 2023.
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
#ifndef _ATATIMEREVENTSOURCE_H
#define _ATATIMEREVENTSOURCE_H


#include <IOKit/IOTypes.h>
#include <IOKit/IOCommandGate.h>
#include <IOKit/IOService.h>
#include <IOKit/IOWorkLoop.h>
#include <IOKit/IOTimerEventSource.h>
#include <IOKit/ndrvsupport/IOMacOSTypes.h>


/*!
@class ATATimerEventSource

@discussion
Extend the timer event source to allow checking for timer expiration 
from behind the workloop.
*/

class ATATimerEventSource : public IOTimerEventSource
{
    OSDeclareDefaultStructors(ATATimerEventSource);

	public:
		
	/*!@function ataTimerEventSource
	@abstract  allocate an instance of this type.
	*/
    static ATATimerEventSource *
	ataTimerEventSource(OSObject *owner, Action action = 0);
	
	/*!@function hasTimedOut
	@abstract returns true if the timer has expired since the last enable/disable or setTimeout() or wakeAtTime() call.
	*/
	virtual bool hasTimedOut( void );
		
	// override to initialize the time out flag.
  	/*!@function 
	@abstract 
	*/
	virtual bool init(OSObject *owner, Action action = 0);

	/*!@function enable
	@abstract overrides in order to set/clear the timed out flag
	*/
	virtual void enable();

	/*!@function disable
	@abstract overrides in order to set/clear the timed out flag
	*/
	virtual void disable();

	/*!@function wakeAtTime
	@abstract overrides in order to set/clear the timed out flag
	*/
	virtual IOReturn wakeAtTime(UnsignedWide abstime);

	/*!@function cancelTimeout
	@abstract overrides in order to set/clear the timed out flag
	*/
	virtual void cancelTimeout();

protected:
	
	enum{  kTimedOutTrue = 'true',
			kTimedOutFalse = 'fals'
		};
	
	UInt32 hasExpired;


	/*!@function myTimeout
	@abstract my timeout function which sets the timedOut flag atomically.
	*/
	static void myTimeout(void *self);

	/*!@function setTimeoutFunc
	@abstract override to install my timeout function instead of the super's.
	*/
    virtual void setTimeoutFunc();

	/*! @struct ExpansionData
    @discussion This structure will be used to expand the capablilties of the IOWorkLoop in the future.
    */
    struct ExpansionData { };

	/*! @var reserved
    Reserved for future use.  (Internal use only)  */
    ExpansionData *reserved;

private:
    OSMetaClassDeclareReservedUnused(ATATimerEventSource, 0);
    OSMetaClassDeclareReservedUnused(ATATimerEventSource, 1);
    OSMetaClassDeclareReservedUnused(ATATimerEventSource, 2);
    OSMetaClassDeclareReservedUnused(ATATimerEventSource, 3);
    OSMetaClassDeclareReservedUnused(ATATimerEventSource, 4);
    OSMetaClassDeclareReservedUnused(ATATimerEventSource, 5);
    OSMetaClassDeclareReservedUnused(ATATimerEventSource, 6);
    OSMetaClassDeclareReservedUnused(ATATimerEventSource, 7);
	
};


#endif /*_ATATIMEREVENTSOURCE_H*/
