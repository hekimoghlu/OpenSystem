/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 8, 2025.
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
 * 24 Nov  1998 suurballe  Created.
 */
#ifndef _IORTCCONTROLLER_H
#define _IORTCCONTROLLER_H

#include <IOKit/IOService.h>

typedef void (*RTC_tick_handler)( IOService * );


class IORTCController : public IOService
{
	OSDeclareAbstractStructors(IORTCController);

public:

	virtual IOReturn getRealTimeClock( UInt8 * currentTime, IOByteCount * length ) = 0;
	virtual IOReturn setRealTimeClock( UInt8 * newTime ) = 0;
};

class IORTC : public IOService
{
	OSDeclareAbstractStructors(IORTC);

protected:

/*! @var reserved
 *   Reserved for future use.  (Internal use only)  */
	struct ExpansionData { };
	ExpansionData *iortc_reserved __unused;

public:

	virtual long            getGMTTimeOfDay( void ) = 0;
	virtual void            setGMTTimeOfDay( long secs ) = 0;

	virtual void                    getUTCTimeOfDay( clock_sec_t * secs, clock_nsec_t * nsecs );
	virtual void                    setUTCTimeOfDay( clock_sec_t secs, clock_nsec_t nsecs );

	virtual void            setAlarmEnable( IOOptionBits message ) = 0;

	virtual IOReturn        getMonotonicClockOffset( int64_t * usecs );
	virtual IOReturn        setMonotonicClockOffset( int64_t usecs );
	virtual IOReturn        getMonotonicClockAndTimestamp( uint64_t * usecs, uint64_t *mach_absolute_time );


	OSMetaClassDeclareReservedUnused(IORTC, 0);
	OSMetaClassDeclareReservedUnused(IORTC, 1);
	OSMetaClassDeclareReservedUnused(IORTC, 2);
	OSMetaClassDeclareReservedUnused(IORTC, 3);
	OSMetaClassDeclareReservedUnused(IORTC, 4);
	OSMetaClassDeclareReservedUnused(IORTC, 5);
	OSMetaClassDeclareReservedUnused(IORTC, 6);
	OSMetaClassDeclareReservedUnused(IORTC, 7);
};

#endif /* !_IORTCCONTROLLER_H */
