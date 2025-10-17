/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 23, 2021.
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
#ifndef __DEBUG_SUPPORT_H__
#define __DEBUG_SUPPORT_H__


#if KERNEL
#include <IOKit/IOLib.h>
#else
#include <IOKit/IOKitLib.h>
#endif


#ifndef DEBUG_ASSERT_COMPONENT_NAME_STRING
	#define DEBUG_ASSERT_COMPONENT_NAME_STRING "AppleSCSIEmulator"
#endif

#if KERNEL

#ifdef __cplusplus
extern "C" {
#endif

/* AppleSCSIEmulatorDebugAssert prototype*/
void
AppleSCSIEmulatorDebugAssert (
						const char * componentNameString,
						const char * assertionString, 
						const char * exceptionLabelString,
						const char * errorString,
						const char * fileName,
						long lineNumber,
						int errorCode );

#ifdef __cplusplus
}
#endif


#define DEBUG_ASSERT_MESSAGE( componentNameString, 					\
	assertionString, 												\
	exceptionLabelString, 											\
	errorString, 													\
	fileName, 														\
	lineNumber, 													\
	error ) 														\
	AppleSCSIEmulatorDebugAssert( componentNameString,				\
	assertionString, 												\
	exceptionLabelString, 											\
	errorString, 													\
	fileName, 														\
	lineNumber, 													\
	error )


#endif	/* KERNEL */


#include <AssertMacros.h>


#define require_success( errorCode, exceptionLabel ) \
	require( kIOReturnSuccess == (errorCode), exceptionLabel )

#define require_success_action( errorCode, exceptionLabel, action ) \
	require_action( kIOReturnSuccess == (errorCode), exceptionLabel, action )

#define require_success_quiet( errorCode, exceptionLabel ) \
	require_quiet( kIOReturnSuccess == (errorCode), exceptionLabel )

#define require_success_action_quiet( errorCode, exceptionLabel, action ) \
	require_action_quiet( kIOReturnSuccess == (errorCode), exceptionLabel, action )

#define require_success_string( errorCode, exceptionLabel, message ) \
	require_string( kIOReturnSuccess == (errorCode), exceptionLabel, message )

#define require_success_action_string( errorCode, exceptionLabel, action, message ) \
	require_action_string( kIOReturnSuccess == (errorCode), exceptionLabel, action, message )


#define require_nonzero( obj, exceptionLabel ) \
	require( ( 0 != obj ), exceptionLabel )

#define require_nonzero_action( obj, exceptionLabel, action ) \
	require_action( ( 0 != obj ), exceptionLabel, action )

#define require_nonzero_quiet( obj, exceptionLabel ) \
	require_quiet( ( 0 != obj ), exceptionLabel )

#define require_nonzero_action_quiet( obj, exceptionLabel, action ) \
	require_action_quiet( ( 0 != obj ), exceptionLabel, action )

#define require_nonzero_string( obj, exceptionLabel, message ) \
	require_string( ( 0 != obj ), exceptionLabel, message )

#define require_nonzero_action_string( obj, exceptionLabel, action, message ) \
	require_action_string( ( 0 != obj ), exceptionLabel, action, message )


#endif	/* __DEBUG_SUPPORT_H__ */
