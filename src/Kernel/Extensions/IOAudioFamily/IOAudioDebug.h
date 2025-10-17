/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 10, 2021.
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
#ifndef _IOAUDIODEBUG_H
#define _IOAUDIODEBUG_H

#include <IOKit/IOTypes.h>
#include <sys/kdebug.h>


#ifdef DEBUG
	#define DEBUG_LEVEL 4						//<rdar://problem/9725460>
	//#define DEBUG_USE_FIRELOG 1
	#define DEBUG_USE_FIREWIRE_KPRINTF 1
	
	#ifdef DEBUG_USE_FIRELOG
	#include <IOKit/firewire/FireLog.h>
	#define audioDebugIOLog( level, message... ) \
		do {FireLog(  message ); FireLog("\n");} while (0)
    #define audioErrorIOLog( message... ) \
        do { FireLog( message ); FireLog("\n"); IOLog( message );} while (0)
	#endif

	#ifdef DEBUG_USE_IOUSBLOG
	#include <IOKit/usb/IOUSBLog.h>
	#define audioDebugIOLog( level, message... ) \
		do {USBLog( level, message );} while (0)
    #define audioErrorIOLog( level, message... ) \
        do { USBLog( DEBUG_LEVEL_BETA, message ); IOLog( message );} while (0)
	#endif

	#ifdef DEBUG_USE_FIREWIRE_KPRINTF
	#define audioDebugIOLog( level, message... ) \
		do { if (level <= DEBUG_LEVEL) kprintf( message );} while (0)
    #define audioErrorIOLog( message... ) \
        do { kprintf( message ); IOLog( message );} while (0)
	#endif

	#ifdef assert
		#undef assert

		#define AssertionMessage( cond, file, line ) \
			"assert \"" #cond "\" failed in " #file " at line " #line

		#define AssertionFailed( cond, file, line ) \
			panic(AssertionMessage( cond, file, line ));

		#define	assert( cond )								\
			if( !(cond) ) {									\
				AssertionFailed( cond, __FILE__, __LINE__ )	\
			}
	#endif
#else
    #define DEBUG_LEVEL 3

    #include <os/log.h>
    #define audioDebugIOLog( level, message... ) \
        do { if ( __builtin_expect(level <= DEBUG_LEVEL, 0) ) { os_log( OS_LOG_DEFAULT, message ); } } while (0)

    #define audioErrorIOLog( message... ) \
        do { os_log_error( OS_LOG_DEFAULT, message ); IOLog( message );} while (0)

#endif


#endif /* _IOAUDIODEBUG_H */
