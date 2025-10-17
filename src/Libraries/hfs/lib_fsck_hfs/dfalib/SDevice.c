/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 9, 2023.
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
#include "SRuntime.h"
#include "check.h"

#if BSD

#include <unistd.h>
#include <errno.h>
#include <sys/ioctl.h>

#include <IOKit/storage/IOMediaBSDClient.h>

#else

#include <Files.h>
#include <Device.h>
#include <Disks.h>

#endif


OSErr GetDeviceSize(int driveRefNum, UInt64 *numBlocks, UInt32 *blockSize)
{
#if BSD
	UInt64 devBlockCount = 0;
	int devBlockSize = 0;

    if (state.blockCount == 0) {
		if (state.debug) fsck_print(ctx, LOG_TYPE_INFO, "%s: Device block count was not initialized by user\n",state.cdevname);
		return (-1);
	}
    devBlockCount = state.blockCount;
	
    if (state.devBlockSize == -1) {
		if (state.debug) fsck_print(ctx, LOG_TYPE_INFO, "%s: Device block size was not initialized by user\n", state.cdevname);
		return (-1);
	}
    devBlockSize = state.devBlockSize;

	if (devBlockSize != 512) {
		*numBlocks = (devBlockCount * (UInt64)devBlockSize) / 512;
		*blockSize = 512;
	} else {
		*numBlocks = devBlockCount;
		*blockSize = devBlockSize;
	}
	return (0);
#else
	/* Various Mac OS device constants */
	enum
	{
		/* return format list status code */
		kFmtLstCode = 6,
		
		/* reference number of .SONY driver */
		kSonyRefNum = 0xfffb,
		
		/* values returned by DriveStatus in DrvSts.twoSideFmt */
		kSingleSided = 0,
		kDoubleSided = -1,
		kSingleSidedSize = 800,		/* 400K */
		kDoubleSidedSize = 1600,	/* 800K */
		
		/* values in DrvQEl.qType */
		kWordDrvSiz = 0,
		kLongDrvSiz = 1,
		
		/* more than enough formatListRecords */
		kMaxFormatListRecs = 16
	};
	
	ParamBlockRec	pb;
	FormatListRec	formatListRecords[kMaxFormatListRecs];
	DrvSts			status;
	short			formatListRecIndex;
	OSErr			result;
	unsigned long	blocks			= 0;

	
	/* Attempt to get the drive's format list. */
	/* (see the Technical Note "What Your Sony Drives For You") */
	
	pb.cntrlParam.ioVRefNum = driveQElementPtr->dQDrive;
	pb.cntrlParam.ioCRefNum = driveQElementPtr->dQRefNum;
	pb.cntrlParam.csCode = kFmtLstCode;
	pb.cntrlParam.csParam[0] = kMaxFormatListRecs;
	*(long *)&pb.cntrlParam.csParam[1] = (long)&formatListRecords[0];
	
	result = PBStatusSync(&pb);
	
	if ( result == noErr )
	{
		/* The drive supports ReturnFormatList status call. */
		
		/* Get the current disk's size. */
		for( formatListRecIndex = 0;
			 formatListRecIndex < pb.cntrlParam.csParam[0];
			 ++formatListRecIndex )
		{
			if ( (formatListRecords[formatListRecIndex].formatFlags &
				  diCIFmtFlagsCurrentMask) != 0 )
			{
				blocks = formatListRecords[formatListRecIndex].volSize;
			}
		}
		if ( blocks == 0 )
		{
			/* This should never happen */
			result = paramErr;
		}
	}
	else if ( driveQElementPtr->dQRefNum == (short)kSonyRefNum )
	{
		/* The drive is a non-SuperDrive floppy which only supports 400K and 800K disks */
		
		result = DriveStatus(driveQElementPtr->dQDrive, &status);
		if ( result == noErr )
		{
			switch ( status.twoSideFmt )
			{
				case kSingleSided:
					blocks = kSingleSidedSize;
					break;
					
				case kDoubleSided:
					blocks = kDoubleSidedSize;
					break;
					
				default:		//	This should never happen
					result = paramErr;
					break;
			}
		}
	}
	else
	{
		/* The drive is not a floppy and it doesn't support ReturnFormatList */
		/* so use the dQDrvSz field(s) */
		
		result = noErr;	/* reset result */
		
		switch ( driveQElementPtr->qType )
		{
			case kWordDrvSiz:
				blocks = driveQElementPtr->dQDrvSz;
				break;
				
			case kLongDrvSiz:
				blocks = ((unsigned long)driveQElementPtr->dQDrvSz2 << 16) +
						 driveQElementPtr->dQDrvSz;
				break;
				
			default:		//	This should never happen
				result = paramErr;
				break;
		}
	}

	*numBlocks = blocks;
	*blockSize = 512;
	
	return( result );
#endif
}


OSErr DeviceRead(int device, int drive, void* buffer, SInt64 offset, UInt32 reqBytes, UInt32 *actBytes)
{
#if BSD
	off_t seek_off;
	ssize_t	nbytes;
	
	*actBytes = 0;

	seek_off = lseek(device, offset, SEEK_SET);
	if (seek_off == -1) {
		if (state.debug) fsck_print(ctx, LOG_TYPE_INFO, "# DeviceRead: lseek(%qd) failed with %d\n", offset, errno);
		return (errno);
	}

	nbytes = read(device, buffer, reqBytes);
	if (nbytes == -1)
		return (errno);
	if (nbytes == 0) {
		if (state.debug) fsck_print(ctx, LOG_TYPE_INFO, "CANNOT READ: BLK %ld\n", (long)offset/512);
		return (5);
	}

	*actBytes = (UInt32)nbytes;
	return (0);

#else
	OSErr err;
	XIOParam pb;

	pb.ioVRefNum	= drive;
	pb.ioRefNum	= device;
	pb.ioPosMode	= fsFromStart;
	pb.ioReqCount	= reqBytes;
	pb.ioBuffer	= buffer;

	if ( (offset & 0xFFFFFFFF00000000) != 0 )
	{
		*(SInt64*)&pb.ioWPosOffset = offset;
		pb.ioPosMode |= (1 << kWidePosOffsetBit);
	}
	else
	{
		((IOParam*)&pb)->ioPosOffset = offset;
	}

	err = PBReadSync( (ParamBlockRec *)&pb );

	return (err);
#endif
}


OSErr DeviceWrite(int device, int drive, void* buffer, SInt64 offset, UInt32 reqBytes, UInt32 *actBytes)
{
#if BSD
	off_t seek_off;
	ssize_t	nbytes;

	*actBytes = 0;

	seek_off = lseek(device, offset, SEEK_SET);
	if (seek_off == -1) {
		if (state.debug) fsck_print(ctx, LOG_TYPE_INFO, "# DeviceRead: lseek(%qd) failed with %d\n", offset, errno);
		return (errno);
	}

	nbytes = write(device, buffer, reqBytes);
	if (nbytes == -1) {
		return (errno);
	}
	if (nbytes == 0) {
		if (state.debug) fsck_print(ctx, LOG_TYPE_INFO, "CANNOT WRITE: BLK %ld\n", (long)offset/512);
		return (5);
	}

	*actBytes = (UInt32)nbytes;
	return (0);
#else
	OSErr err;
	XIOParam pb;

	pb.ioVRefNum	= drive;
	pb.ioRefNum	= device;
	pb.ioPosMode	= fsFromStart;
	pb.ioReqCount	= reqBytes;
	pb.ioBuffer	= buffer;

	if ( (offset & 0xFFFFFFFF00000000) != 0 )
	{
		*(SInt64*)&pb.ioWPosOffset = offset;
		pb.ioPosMode |= (1 << kWidePosOffsetBit);
	}
	else
	{
		((IOParam*)&pb)->ioPosOffset = offset;
	}

	err = PBWriteSync( (ParamBlockRec *)&pb );

	return (err);
#endif
}
