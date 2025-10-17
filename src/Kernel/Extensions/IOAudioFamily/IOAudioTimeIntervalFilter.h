/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 21, 2023.
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
#ifndef _IOAUDIOTIMEINTERVALFILTER_H
#define _IOAUDIOTIMEINTERVALFILTER_H

#include "BigNum128.h"

/*!
 @class IOAudioTimeIntervalFilter
 @abstract An abstract class that provides a filtered timeline based on snapshots from jittery time captures
 */
class IOAudioTimeIntervalFilter : public OSObject
{
	OSDeclareAbstractStructors(IOAudioTimeIntervalFilter)

public:
	/*!
	 @function reInitialiseFilter
	 @abstract Restart a new timeline sequence, with a new expected interval spacing
	 @param expectedInterval Expected interval of time captures. Pass zero to use the results from previous runs.
	 @param multiIntervalCount Count of multiple intervals to return from getMultiIntervalTime.
	 */
	virtual IOReturn reInitialiseFilter(uint32_t expectedInterval = 0, uint32_t multiIntervalCount = 1 );

    /*!
     * @function free
     * @abstract Frees all of the resources allocated by the IOAudioTimeIntervalFilter.
     * @discussion Do not call this directly.  This is called automatically by the system when the instance's
     *  refcount goes to 0.  To decrement the refcount, call release() on the object.
     */
    virtual void free();



	/*!
	 @function newTimePosition
	 @abstract Pass in the raw measured time position
	 @param rawSnapshot The raw time position. These should be approximately occurring every ExpectedInterval
	 @result A filtered time position
	 */

	virtual AbsoluteTime newTimePosition(AbsoluteTime rawSnapshot);

	/*!
	 @function getMultiIntervalTime
	 @abstract Return the time between the last MultiIntervalCount intervals of the filtered timeline
	 @result Return the time between the last MultiIntervalCount intervals of the filtered timeline
	 */
	virtual uint64_t getMultiIntervalTime(void);

	
	OSMetaClassDeclareReservedUnused (IOAudioTimeIntervalFilter, 0 );
	OSMetaClassDeclareReservedUnused (IOAudioTimeIntervalFilter, 1 );
	OSMetaClassDeclareReservedUnused (IOAudioTimeIntervalFilter, 2 );
	OSMetaClassDeclareReservedUnused (IOAudioTimeIntervalFilter, 3 );
	OSMetaClassDeclareReservedUnused (IOAudioTimeIntervalFilter, 4 );
	OSMetaClassDeclareReservedUnused (IOAudioTimeIntervalFilter, 5 );	
	OSMetaClassDeclareReservedUnused (IOAudioTimeIntervalFilter, 6 );
	OSMetaClassDeclareReservedUnused (IOAudioTimeIntervalFilter, 7 );
	OSMetaClassDeclareReservedUnused (IOAudioTimeIntervalFilter, 8 );
	OSMetaClassDeclareReservedUnused (IOAudioTimeIntervalFilter, 9 );
	OSMetaClassDeclareReservedUnused (IOAudioTimeIntervalFilter, 10 );
	OSMetaClassDeclareReservedUnused (IOAudioTimeIntervalFilter, 11 );	
	OSMetaClassDeclareReservedUnused (IOAudioTimeIntervalFilter, 12 );
	OSMetaClassDeclareReservedUnused (IOAudioTimeIntervalFilter, 13 );
	OSMetaClassDeclareReservedUnused (IOAudioTimeIntervalFilter, 14 );
	OSMetaClassDeclareReservedUnused (IOAudioTimeIntervalFilter, 15 );

protected:
	/* <rdar://12136103> */
    struct ExpansionData
	{
	};
    
    ExpansionData   *reserved;
	
	/*!
	 @function initFilter
	 @abstract Construct a new instance of the TimeFilter class
	 @param ExpectedInterval Expected interval of time captures
	 @param MultiIntervalCount Optionally calculate the count between ExpectedInterval
	 */
	virtual bool initFilter(uint32_t expectedInterval, uint32_t multiIntervalCount = 1);

	/*!
	 @function calculateNewTimePosition
	 @abstract abstract method to calculate the new time position based on the raw snapshot
	 @param rawSnapshot Raw filter value
	 @result filtered time value
	 */
	virtual uint64_t calculateNewTimePosition(uint64_t rawSnapshot) = 0;

	inline int decCircularBufferPosition(int n, int dec = 1)	{ return (n + mMultiIntervalCount - dec) % mMultiIntervalCount;  }
	inline int incCircularBufferPosition(int n, int inc = 1)	{ return (n + mMultiIntervalCount + inc) % mMultiIntervalCount;  }

	uint32_t	mExpectedInterval;
	uint32_t	mMultiIntervalCount;
	
	uint64_t*	mIntervalTimeHistory;
	
    /*!
     * @var mIntervalTimeHistoryPointer 
     *  Points to the next time interval to be updated
     */
	int			mIntervalTimeHistoryPointer;

    /*!
     * @var mFilterCount 
     *  How many times the filter has been called since re-init
     */
	uint64_t	mFilterCount;
	
    IOLock*		timeIntervalLock;
};



/*!
 @class IOAudioTimeIntervalFilterIIR
 @abstract A concrete IOAudioTimeIntervalFilter class that provides an IIR-based filtered timeline based on snapshots from jittery time captures
 */

class IOAudioTimeIntervalFilterIIR : public IOAudioTimeIntervalFilter
{
    OSDeclareDefaultStructors(IOAudioTimeIntervalFilterIIR)

public:
	/*!
	 @function initFilter
	 @abstract Construct a new instance of the IIR TimeFilter class
	 @param ExpectedInterval Expected interval of time captures
	 @param MultiIntervalCount Optionally calculate the count between ExpectedInterval
	 @param filterCoef IIR filter coefficient. Increase this number for more aggressive smoothing
	 */
	virtual bool initFilter(uint32_t expectedInterval, uint32_t multiIntervalCount = 1, uint16_t filterCoef = 4);

	OSMetaClassDeclareReservedUnused (IOAudioTimeIntervalFilterIIR, 0 );
	OSMetaClassDeclareReservedUnused (IOAudioTimeIntervalFilterIIR, 1 );
	OSMetaClassDeclareReservedUnused (IOAudioTimeIntervalFilterIIR, 2 );
	OSMetaClassDeclareReservedUnused (IOAudioTimeIntervalFilterIIR, 3 );
	OSMetaClassDeclareReservedUnused (IOAudioTimeIntervalFilterIIR, 4 );
	OSMetaClassDeclareReservedUnused (IOAudioTimeIntervalFilterIIR, 5 );	
	OSMetaClassDeclareReservedUnused (IOAudioTimeIntervalFilterIIR, 6 );
	OSMetaClassDeclareReservedUnused (IOAudioTimeIntervalFilterIIR, 7 );
	OSMetaClassDeclareReservedUnused (IOAudioTimeIntervalFilterIIR, 8 );
	OSMetaClassDeclareReservedUnused (IOAudioTimeIntervalFilterIIR, 9 );
	OSMetaClassDeclareReservedUnused (IOAudioTimeIntervalFilterIIR, 10 );
	OSMetaClassDeclareReservedUnused (IOAudioTimeIntervalFilterIIR, 11 );	
	OSMetaClassDeclareReservedUnused (IOAudioTimeIntervalFilterIIR, 12 );
	OSMetaClassDeclareReservedUnused (IOAudioTimeIntervalFilterIIR, 13 );
	OSMetaClassDeclareReservedUnused (IOAudioTimeIntervalFilterIIR, 14 );
	OSMetaClassDeclareReservedUnused (IOAudioTimeIntervalFilterIIR, 15 );

protected:
	virtual void IIR(U128* filterVal, U128 input, int shiftAmount);
	virtual uint64_t calculateNewTimePosition(uint64_t rawSnapshot);
	
	U128		mFilteredSnapshot;
	U128		mFilteredOffset;
	uint16_t	mIIRCoef;
};



/*!
 @class IOAudioTimeIntervalFilterFIR
 @abstract A concrete IOAudioTimeIntervalFilter class that provides an FIR-based filtered timeline based on snapshots from jittery time captures
 */

class IOAudioTimeIntervalFilterFIR : public IOAudioTimeIntervalFilter
{
    OSDeclareDefaultStructors(IOAudioTimeIntervalFilterFIR)
	
public:
	/*!
	 @function initFilter
	 @abstract Construct a new instance of the IIR TimeFilter class
	 @param ExpectedInterval Expected interval of time captures
	 @param MultiIntervalCount Optionally calculate the count between ExpectedInterval
	 */
	virtual bool initFilter(uint32_t expectedInterval, uint32_t multiIntervalCount = 1);

    /*!
     * @function free
     * @abstract Frees all of the resources allocated by the IOAudioTimeIntervalFilter.
     * @discussion Do not call this directly.  This is called automatically by the system when the instance's
     *  refcount goes to 0.  To decrement the refcount, call release() on the object.
     */
    virtual void free();
	
	IOReturn reInitialiseFilter(uint32_t expectedInterval = 0, uint32_t multiIntervalCount = 1 );
	
	OSMetaClassDeclareReservedUnused (IOAudioTimeIntervalFilterFIR, 0 );
	OSMetaClassDeclareReservedUnused (IOAudioTimeIntervalFilterFIR, 1 );
	OSMetaClassDeclareReservedUnused (IOAudioTimeIntervalFilterFIR, 2 );
	OSMetaClassDeclareReservedUnused (IOAudioTimeIntervalFilterFIR, 3 );
	OSMetaClassDeclareReservedUnused (IOAudioTimeIntervalFilterFIR, 4 );
	OSMetaClassDeclareReservedUnused (IOAudioTimeIntervalFilterFIR, 5 );	
	OSMetaClassDeclareReservedUnused (IOAudioTimeIntervalFilterFIR, 6 );
	OSMetaClassDeclareReservedUnused (IOAudioTimeIntervalFilterFIR, 7 );
	OSMetaClassDeclareReservedUnused (IOAudioTimeIntervalFilterFIR, 8 );
	OSMetaClassDeclareReservedUnused (IOAudioTimeIntervalFilterFIR, 9 );
	OSMetaClassDeclareReservedUnused (IOAudioTimeIntervalFilterFIR, 10 );
	OSMetaClassDeclareReservedUnused (IOAudioTimeIntervalFilterFIR, 11 );	
	OSMetaClassDeclareReservedUnused (IOAudioTimeIntervalFilterFIR, 12 );
	OSMetaClassDeclareReservedUnused (IOAudioTimeIntervalFilterFIR, 13 );
	OSMetaClassDeclareReservedUnused (IOAudioTimeIntervalFilterFIR, 14 );
	OSMetaClassDeclareReservedUnused (IOAudioTimeIntervalFilterFIR, 15 );
protected:

	virtual uint64_t calculateNewTimePosition(uint64_t rawSnapshot);
	virtual IOReturn setNewFilter(uint32_t numCoeffs, const uint64_t* filterCoefficients, uint32_t scale);

	U128 FIR(uint64_t *history, uint64_t input);

	uint64_t*	mCoeffs;
	uint64_t*	mDataOffsetHistory;
	uint64_t*	mDataHistory;
	uint32_t	mNumCoeffs;
	uint32_t	mFilterScale;
	uint32_t	mFilterWritePointer;
};

#endif		// _IOAUDIOTIMEINTERVALFILTER_H

