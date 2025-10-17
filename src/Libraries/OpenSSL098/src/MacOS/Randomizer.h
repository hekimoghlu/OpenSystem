/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 7, 2024.
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


//	Gathers unpredictable system data to be used for generating
//	random bits

#include <MacTypes.h>

class CRandomizer
{
public:
	CRandomizer (void);
	void PeriodicAction (void);
	
private:

	// Private calls

	void		AddTimeSinceMachineStartup (void);
	void		AddAbsoluteSystemStartupTime (void);
	void		AddAppRunningTime (void);
	void		AddStartupVolumeInfo (void);
	void		AddFiller (void);

	void		AddCurrentMouse (void);
	void		AddNow (double millisecondUncertainty);
	void		AddBytes (void *data, long size, double entropy);
	
	void		GetTimeBaseResolution (void);
	unsigned long	SysTimer (void);

	// System Info	
	bool		mSupportsLargeVolumes;
	bool		mIsPowerPC;
	bool		mIs601;
	
	// Time info
	double		mTimebaseTicksPerMillisec;
	unsigned long	mLastPeriodicTicks;
	
	// Mouse info
	long		mSamplePeriod;
	Point		mLastMouse;
	long		mMouseStill;
};
