/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 28, 2024.
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
// AppleCDDAFileSystemUtils.h created by CJS on Sun 14-May-2000

#ifndef __APPLE_CDDA_FS_UTILS_H__
#define __APPLE_CDDA_FS_UTILS_H__

#ifndef __APPLE_CDDA_FS_DEFINES_H__
#include "AppleCDDAFileSystemDefines.h"
#endif

#include <sys/vnode.h>
#include <sys/attr.h>

#ifdef __cplusplus
extern "C" {
#endif


//-----------------------------------------------------------------------------
//	Function Prototypes - From AppleCDDAFileSystemUtils.c
//-----------------------------------------------------------------------------

int				InsertCDDANode 				( AppleCDDANodePtr newNodePtr,
											  vnode_t parentVNodePtr,
											  struct proc * theProcPtr );
errno_t			CreateNewCDDANode 			( mount_t mountPtr,
											  UInt32 nodeID,
											  enum vtype vNodeType,
											  vnode_t parentVNodePtr,
											  struct componentname * compNamePtr,
											  vnode_t * vNodeHandle );
int				DisposeCDDANode 			( vnode_t vNodePtr );
errno_t			CreateNewCDDAFile 			( mount_t mountPtr,
											  UInt32 nodeID,
											  AppleCDDANodeInfoPtr nodeInfoPtr,
											  vnode_t parentVNodePtr,
											  struct componentname * compNamePtr,
											  vnode_t * vNodeHandle );
errno_t			CreateNewXMLFile 			( mount_t mountPtr,
											  UInt32 xmlFileSize,
											  UInt8 * xmlData,
											  vnode_t parentVNodePtr,
											  struct componentname * compNamePtr,
											  vnode_t * vNodeHandle );
errno_t			CreateNewCDDADirectory 		( mount_t mountPtr,
											  UInt32 nodeID,
											  vnode_t * vNodeHandle );
boolean_t		IsAudioTrack 				( const SubQTOCInfoPtr trackDescriptorPtr );
UInt32			CalculateSize 				( const QTOCDataFormat10Ptr TOCDataPtr,
											  UInt32 trackDescriptorOffset,
											  UInt32 currentA2Offset );
SInt32			ParseTOC 					( mount_t mountPtr,
											  UInt32 numTracks );
int				GetTrackNumberFromName 		( const char * name,
											  UInt32 * trackNumber );

int				CalculateAttributeBlockSize	( struct attrlist * attrlist );
void			PackAttributesBlock			( struct attrlist * attrListPtr,
											  vnode_t vNodePtr,
											  void ** attrbufHandle,
											  void ** varbufHandle );


//-----------------------------------------------------------------------------
//	Function Prototypes - From AppleCDDAFileSystemUtilities.cpp
//-----------------------------------------------------------------------------


QTOCDataFormat10Ptr		CreateBufferFromIORegistry 	( mount_t mountPtr );
void					DisposeBufferFromIORegistry	( QTOCDataFormat10Ptr TOCDataPtr );


#ifdef __cplusplus
}
#endif

#endif // __APPLE_CDDA_FS_UTILS_H__
