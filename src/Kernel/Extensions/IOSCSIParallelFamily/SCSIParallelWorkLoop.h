/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 3, 2024.
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
#ifndef __IOKIT_SCSI_PARALLEL_WORKLOOP_H__
#define __IOKIT_SCSI_PARALLEL_WORKLOOP_H__


//-----------------------------------------------------------------------------
//	Includes
//-----------------------------------------------------------------------------

// General IOKit includes
#include <IOKit/IOWorkLoop.h>
#include <IOKit/IOLocksPrivate.h>

//-----------------------------------------------------------------------------
//	Class Declarations
//-----------------------------------------------------------------------------

class SCSIParallelWorkLoop : public IOWorkLoop
{
	
	OSDeclareDefaultStructors ( SCSIParallelWorkLoop )
	
public:
	
	static SCSIParallelWorkLoop *	Create ( const char * lockGroupName );
	
	bool	InitWithLockGroupName ( const char * lockGroupName );
	void	free ( void );
	
	lck_grp_t *		fLockGroup;
	
};


#endif	/* __IOKIT_SCSI_PARALLEL_WORKLOOP_H__ */