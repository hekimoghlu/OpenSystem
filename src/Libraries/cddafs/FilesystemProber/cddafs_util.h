/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 30, 2024.
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
// cddafs_util.h created by CJS on Mon 18-May-2000

#ifndef __CDDAFS_UTIL_H__
#define __CDDAFS_UTIL_H__

#include "AppleCDDAFileSystemDefines.h"

#ifdef __cplusplus
extern "C" {
#endif


//-----------------------------------------------------------------------------
//	Constants
//-----------------------------------------------------------------------------

// cddafs Stuff
#define kCDDAFileSystemName				"cddafs"
#define	kMountPointName					"Audio CD"
#define kCDDAFileSystemMountType		"cddafs"

// (Un)Mount Stuff
#define kMountCommand					"/sbin/mount"
#define kUnmountCommand					"/sbin/umount"
#define kLoadCommand 					"/sbin/kextload"
#define kCDDAFileSystemExtensionPath 	"/System/Library/Extensions/cddafs.kext"

#define kMountExecutableName			"/sbin/mount_cddafs"
#define kUtilExecutableName				"cddafs.util"

#define kIOCDMediaTOC					"TOC"
#define kIOCDMediaString				"IOCDMedia"

#ifndef FSUR_MOUNT_HIDDEN
#define FSUR_MOUNT_HIDDEN (-9)
#endif

/* XML PList keys */
#define kRawTOCDataString				"Format 0x02 TOC Data"
#define kSessionsString					"Sessions"
#define kSessionTypeString				"Session Type"
#define kTrackArrayString				"Track Array"
#define kFirstTrackInSessionString		"First Track"
#define kLastTrackInSessionString		"Last Track"
#define kLeadoutBlockString				"Leadout Block"
#define	kDataString 					"Data"
#define kPointString					"Point"
#define kSessionNumberString			"Session Number"
#define kStartBlockString		 		"Start Block"
#define kPreEmphasisString				"Pre-Emphasis Enabled"

enum
{
	kUsageTypeUtility	= 0,
	kUsageTypeMount		= 1
};


//-----------------------------------------------------------------------------
//	Function Prototypes
//-----------------------------------------------------------------------------


void				DisplayUsage  			( int usageType, const char * argv[] );
void 				StripTrailingSpaces		( char * contentsPtr );
void				WriteDiskLabel 			( char * contentsPtr );
int					Mount					( const char * 	deviceNamePtr,
											  const char * 	mountPointPtr,
											  int			mountFlags );
int 				Probe 					( char * deviceNamePtr );
int 				Unmount 				( const char * mountPtPtr );
int					ParseMountArgs			( int * argc, const char ** argv[], int * mountFlags );
int					ParseUtilityArgs 		( int argc,
											  const char * argv[],
											  const char ** actionPtr,
											  const char ** mountPointPtr,
											  boolean_t * isEjectablePtr,
											  boolean_t * isLockedPtr );
UInt8 *				CreateBufferFromCFData 	( CFDataRef string );
CFDataRef			CreateXMLFileInPListFormat ( QTOCDataFormat10Ptr TOCDataPtr );
UInt32				FindNumberOfAudioTracks ( QTOCDataFormat10Ptr TOCDataPtr );

int					ParseTOC 				( UInt8 * TOCInfoPtr );
UInt8 *				GetTOCDataPtr			( const char * deviceNamePtr );
UInt8				GetPointValue			( UInt32 trackIndex, QTOCDataFormat10Ptr TOCData );
Boolean				IsAudioTrack			( UInt32 trackNumber, QTOCDataFormat10Ptr TOCData );
SInt32				GetNumberOfTrackDescriptors ( 	QTOCDataFormat10Ptr	TOCDataPtr,
													UInt8 * 			numberOfDescriptors );

int					GetVFSConfigurationByName ( const char * fileSystemName,
												struct vfsconf * vfsConfPtr );
int					LoadKernelExtension		( void );

#ifdef __cplusplus
}
#endif

#endif // __CDDAFS_UTIL_H__