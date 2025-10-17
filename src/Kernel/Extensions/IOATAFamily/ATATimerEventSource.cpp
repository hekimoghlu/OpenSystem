/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 27, 2023.
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
/*
 * ATATimerEventSource.cpp
 *
 *	implements a timer event source that can be checked from behind the 
 *  workloop for a timeout event.
 */
 
#include <sys/cdefs.h>

__BEGIN_DECLS
#include <kern/thread_call.h>
__END_DECLS

#include <IOKit/assert.h>
#include <IOKit/system.h>

#include <IOKit/IOLib.h>
#include <IOKit/IOTimerEventSource.h>
#include <IOKit/IOWorkLoop.h>
#include "ATATimerEventSource.h"


#define super IOTimerEventSource
OSDefineMetaClassAndStructors(ATATimerEventSource, IOTimerEventSource)
OSMetaClassDefineReservedUnused(ATATimerEventSource, 0);
OSMetaClassDefineReservedUnused(ATATimerEventSource, 1);
OSMetaClassDefineReservedUnused(ATATimerEventSource, 2);
OSMetaClassDefineReservedUnused(ATATimerEventSource, 3);
OSMetaClassDefineReservedUnused(ATATimerEventSource, 4);
OSMetaClassDefineReservedUnused(ATATimerEventSource, 5);
OSMetaClassDefineReservedUnused(ATATimerEventSource, 6);
OSMetaClassDefineReservedUnused(ATATimerEventSource, 7);

// the interesting part of the function.
bool 
ATATimerEventSource::hasTimedOut( void )
{
	return (hasExpired == kTimedOutTrue);
}


// Timeout handler function. This function is called by the kernel when
// the timeout interval expires.
//
void ATATimerEventSource::myTimeout(void *self)
{
    ATATimerEventSource *me = (ATATimerEventSource *) self;
	
	OSCompareAndSwap( kTimedOutFalse, kTimedOutTrue, &(me->hasExpired) );
	
	// pasted from superclass's static handler.
 
	if (me->enabled) 
	{
        Action doit = (Action) me->action;

        if (doit) 
		{
            me->closeGate();
            (*doit)(me->owner, me);
            me->openGate();
        }
    }
}

void ATATimerEventSource::setTimeoutFunc()
{
    calloutEntry = (void *) thread_call_allocate((thread_call_func_t) myTimeout,
                                                 (thread_call_param_t) this);
}

bool ATATimerEventSource::init(OSObject *inOwner, Action inAction)
{
    if ( !super::init( (OSObject *)inOwner, (Action) inAction) )
        return false;

	hasExpired = kTimedOutFalse;
	
    return true;
}


ATATimerEventSource *
ATATimerEventSource::ataTimerEventSource(OSObject *inOwner, Action inAction)
{
    ATATimerEventSource *me = new ATATimerEventSource;

    if (me && !me->init(inOwner, inAction)) {
        me->free();
        return 0;
    }

    return me;
}
void ATATimerEventSource::cancelTimeout()
{
	hasExpired = kTimedOutFalse;
	super::cancelTimeout();
}

void ATATimerEventSource::enable()
{
	hasExpired = kTimedOutFalse;
    super::enable();
}

void ATATimerEventSource::disable()
{
	hasExpired = kTimedOutFalse;

    super::disable();
}

IOReturn ATATimerEventSource::wakeAtTime(UnsignedWide inAbstime)
{
	hasExpired = kTimedOutFalse;
#if ABSOLUTETIME_SCALAR_TYPE
	UInt64	abstime = (UInt64)inAbstime.hi << 32 | (UInt64)inAbstime.lo;
    return super::wakeAtTime( abstime );
#else
	return super::wakeAtTime(inAbstime);
#endif	/* ABSOLUTETIME_SCALAR_TYPE */
}
