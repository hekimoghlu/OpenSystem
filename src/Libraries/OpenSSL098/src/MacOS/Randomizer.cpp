/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 13, 2022.
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
/*------------------------------ Includes ----------------------------*/

#include "Randomizer.h"

// Mac OS API
#include <Files.h>
#include <Folders.h>
#include <Events.h>
#include <Processes.h>
#include <Gestalt.h>
#include <Resources.h>
#include <LowMem.h>

// Standard C library
#include <stdlib.h>
#include <math.h>

/*---------------------- Function declarations -----------------------*/

// declared in OpenSSL/crypto/rand/rand.h
extern "C" void RAND_add (const void *buf, int num, double entropy);

unsigned long GetPPCTimer (bool is601);	// Make it global if needed
					// elsewhere

/*---------------------------- Constants -----------------------------*/

#define kMouseResolution 6		// Mouse position has to differ
					// from the last one by this
					// much to be entered
#define kMousePositionEntropy 5.16	// log2 (kMouseResolution**2)
#define kTypicalMouseIdleTicks 300.0	// I am guessing that a typical
					// amount of time between mouse
					// moves is 5 seconds
#define kVolumeBytesEntropy 12.0	// about log2 (20000/4),
					// assuming a variation of 20K
					// in total file size and
					// long-aligned file formats.
#define kApplicationUpTimeEntropy 6.0	// Variance > 1 second, uptime
					// in ticks  
#define kSysStartupEntropy 7.0		// Entropy for machine startup
					// time


/*------------------------ Function definitions ----------------------*/

CRandomizer::CRandomizer (void)
{
	long	result;
	
	mSupportsLargeVolumes =
		(Gestalt(gestaltFSAttr, &result) == noErr) &&
		((result & (1L << gestaltFSSupports2TBVols)) != 0);
	
	if (Gestalt (gestaltNativeCPUtype, &result) != noErr)
	{
		mIsPowerPC = false;
		mIs601 = false;
	}
	else
	{
		mIs601 = (result == gestaltCPU601);
		mIsPowerPC = (result >= gestaltCPU601);
	}
	mLastMouse.h = mLastMouse.v = -10;	// First mouse will
						// always be recorded
	mLastPeriodicTicks = TickCount();
	GetTimeBaseResolution ();
	
	// Add initial entropy
	AddTimeSinceMachineStartup ();
	AddAbsoluteSystemStartupTime ();
	AddStartupVolumeInfo ();
	AddFiller ();
}

void CRandomizer::PeriodicAction (void)
{
	AddCurrentMouse ();
	AddNow (0.0);	// Should have a better entropy estimate here
	mLastPeriodicTicks = TickCount();
}

/*------------------------- Private Methods --------------------------*/

void CRandomizer::AddCurrentMouse (void)
{
	Point mouseLoc;
	unsigned long lastCheck;	// Ticks since mouse was last
					// sampled

#if TARGET_API_MAC_CARBON
	GetGlobalMouse (&mouseLoc);
#else
	mouseLoc = LMGetMouseLocation();
#endif
	
	if (labs (mLastMouse.h - mouseLoc.h) > kMouseResolution/2 &&
	    labs (mLastMouse.v - mouseLoc.v) > kMouseResolution/2)
		AddBytes (&mouseLoc, sizeof (mouseLoc),
				kMousePositionEntropy);
	
	if (mLastMouse.h == mouseLoc.h && mLastMouse.v == mouseLoc.v)
		mMouseStill ++;
	else
	{
		double entropy;
		
		// Mouse has moved. Add the number of measurements for
		// which it's been still. If the resolution is too
		// coarse, assume the entropy is 0.

		lastCheck = TickCount() - mLastPeriodicTicks;
		if (lastCheck <= 0)
			lastCheck = 1;
		entropy = log2l
			(kTypicalMouseIdleTicks/(double)lastCheck);
		if (entropy < 0.0)
			entropy = 0.0;
		AddBytes (&mMouseStill, sizeof (mMouseStill), entropy);
		mMouseStill = 0;
	}
	mLastMouse = mouseLoc;
}

void CRandomizer::AddAbsoluteSystemStartupTime (void)
{
	unsigned long	now;		// Time in seconds since
					// 1/1/1904
	GetDateTime (&now);
	now -= TickCount() / 60;	// Time in ticks since machine
					// startup
	AddBytes (&now, sizeof (now), kSysStartupEntropy);
}

void CRandomizer::AddTimeSinceMachineStartup (void)
{
	AddNow (1.5);			// Uncertainty in app startup
					// time is > 1.5 msec (for
					// automated app startup).
}

void CRandomizer::AddAppRunningTime (void)
{
	ProcessSerialNumber PSN;
	ProcessInfoRec		ProcessInfo;
	
	ProcessInfo.processInfoLength = sizeof (ProcessInfoRec);
	ProcessInfo.processName = nil;
	ProcessInfo.processAppSpec = nil;
	
	GetCurrentProcess (&PSN);
	GetProcessInformation (&PSN, &ProcessInfo);

	// Now add the amount of time in ticks that the current process
	// has been active

	AddBytes (&ProcessInfo, sizeof (ProcessInfoRec),
			kApplicationUpTimeEntropy);
}

void CRandomizer::AddStartupVolumeInfo (void)
{
	short			vRefNum;
	long			dirID;
	XVolumeParam	pb;
	OSErr			err;
	
	if (!mSupportsLargeVolumes)
		return;
		
	FindFolder (kOnSystemDisk, kSystemFolderType, kDontCreateFolder,
			&vRefNum, &dirID);
	pb.ioVRefNum = vRefNum;
	pb.ioCompletion = 0;
	pb.ioNamePtr = 0;
	pb.ioVolIndex = 0;
	err = PBXGetVolInfoSync (&pb);
	if (err != noErr)
		return;
		
	// Base the entropy on the amount of space used on the disk and
	// on the next available allocation block. A lot else might be
	// unpredictable, so might as well toss the whole block in. See
	// comments for entropy estimate justifications.

	AddBytes (&pb, sizeof (pb),
		kVolumeBytesEntropy +
		log2l (((pb.ioVTotalBytes.hi - pb.ioVFreeBytes.hi)
				* 4294967296.0D +
			(pb.ioVTotalBytes.lo - pb.ioVFreeBytes.lo))
				/ pb.ioVAlBlkSiz - 3.0));
}

/*
	On a typical startup CRandomizer will come up with about 60
	bits of good, unpredictable data. Assuming no more input will
	be available, we'll need some more lower-quality data to give
	OpenSSL the 128 bits of entropy it desires. AddFiller adds some
	relatively predictable data into the soup.
*/

void CRandomizer::AddFiller (void)
{
	struct
	{
		ProcessSerialNumber psn;	// Front process serial
						// number
		RGBColor	hiliteRGBValue;	// User-selected
						// highlight color
		long		processCount;	// Number of active
						// processes
		long		cpuSpeed;	// Processor speed
		long		totalMemory;	// Total logical memory
						// (incl. virtual one)
		long		systemVersion;	// OS version
		short		resFile;	// Current resource file
	} data;
	
	GetNextProcess ((ProcessSerialNumber*) kNoProcess);
	while (GetNextProcess (&data.psn) == noErr)
		data.processCount++;
	GetFrontProcess (&data.psn);
	LMGetHiliteRGB (&data.hiliteRGBValue);
	Gestalt (gestaltProcClkSpeed, &data.cpuSpeed);
	Gestalt (gestaltLogicalRAMSize, &data.totalMemory);
	Gestalt (gestaltSystemVersion, &data.systemVersion);
	data.resFile = CurResFile ();
	
	// Here we pretend to feed the PRNG completely random data. This
	// is of course false, as much of the above data is predictable
	// by an outsider. At this point we don't have any more
	// randomness to add, but with OpenSSL we must have a 128 bit
	// seed before we can start. We just add what we can, without a
	// real entropy estimate, and hope for the best.

	AddBytes (&data, sizeof(data), 8.0 * sizeof(data));
	AddCurrentMouse ();
	AddNow (1.0);
}

//-------------------  LOW LEVEL ---------------------

void CRandomizer::AddBytes (void *data, long size, double entropy)
{
	RAND_add (data, size, entropy * 0.125);	// Convert entropy bits
						// to bytes
}

void CRandomizer::AddNow (double millisecondUncertainty)
{
	long time = SysTimer();
	AddBytes (&time, sizeof (time), log2l (millisecondUncertainty *
			mTimebaseTicksPerMillisec));
}

//----------------- TIMING SUPPORT ------------------

void CRandomizer::GetTimeBaseResolution (void)
{	
#ifdef __powerc
	long speed;
	
	// gestaltProcClkSpeed available on System 7.5.2 and above
	if (Gestalt (gestaltProcClkSpeed, &speed) != noErr)
		// Only PowerPCs running pre-7.5.2 are 60-80 MHz
		// machines.
		mTimebaseTicksPerMillisec =  6000.0D;
	// Assume 10 cycles per clock update, as in 601 spec. Seems true
	// for later chips as well.
	mTimebaseTicksPerMillisec = speed / 1.0e4D;
#else
	// 68K VIA-based machines (see Develop Magazine no. 29)
	mTimebaseTicksPerMillisec = 783.360D;
#endif
}

unsigned long CRandomizer::SysTimer (void)	// returns the lower 32
						// bit of the chip timer
{
#ifdef __powerc
	return GetPPCTimer (mIs601);
#else
	UnsignedWide usec;
	Microseconds (&usec);
	return usec.lo;
#endif
}

#ifdef __powerc
// The timebase is available through mfspr on 601, mftb on later chips.
// Motorola recommends that an 601 implementation map mftb to mfspr
// through an exception, but I haven't tested to see if MacOS actually
// does this. We only sample the lower 32 bits of the timer (i.e. a
// few minutes of resolution)

asm unsigned long GetPPCTimer (register bool is601)
{
	cmplwi	is601, 0	// Check if 601
	bne	_601		// if non-zero goto _601
	mftb  	r3		// Available on 603 and later.
	blr			// return with result in r3
_601:
	mfspr r3, spr5  	// Available on 601 only.
				// blr inserted automatically
}
#endif
