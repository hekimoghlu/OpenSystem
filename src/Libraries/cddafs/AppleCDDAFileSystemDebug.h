/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 24, 2023.
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
// AppleCDDAFileSystemDebug.h created by CJS on Tue 20-Jun-2000

#ifndef __APPLE_CDDA_FS_DEBUG_H__
#define __APPLE_CDDA_FS_DEBUG_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <IOKit/IOLib.h>

// Debug Options
#define DEBUG			0
#define DEBUGLEVEL		0		// 1-5  1=fatal errors only, 5=full debugging

#if DEBUG

	#if DEBUGLEVEL > 2
		#define DebugLog(x)		kprintf	x		// Turn DebugLog() on
	#else
		#define DebugLog(x)						// Turn DebugLog() off
	#endif
	
	#if DEBUGLEVEL > 3
		#define MACH_ASSERT		1				// To turn assert() on		
	#endif

#else

	#define DebugLog(x)							// Turn DebugLog() off
	
#endif


#include <kern/assert.h>		// for assert()


#if DEBUG

	#if DEBUGLEVEL > 5
		#define DebugAssert(x)			( void ) assert	x	// Turn DebugAssert() on
		#ifndef __cplusplus
			#define	unused
		#endif
	#else
		#define DebugAssert(x)								// Turn DebugAssert() off
		#ifndef __cplusplus
			#define	unused				__unused
		#endif
	#endif

#else

	#define DebugAssert(x)						// Turn DebugAssert() off
	#ifndef __cplusplus
		#define	unused					__unused
	#endif
	
#endif

#ifdef __cplusplus
}
#endif


#endif // __APPLE_CDDA_FS_DEBUG_H__
