/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 9, 2022.
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
#ifndef _SEC_TRANSFORM_READ_TRANSFORM_H
#define _SEC_TRANSFORM_READ_TRANSFORM_H

#include <Security/SecTransform.h>

#ifdef __cplusplus
extern "C" {
#endif

CF_ASSUME_NONNULL_BEGIN
CF_IMPLICIT_BRIDGING_ENABLED

/*!
	@header

			The read transform reads bytes from a instance.  The bytes are
			sent as CFDataRef instances to the OUTPUT attribute of the
			transform.
				
			This transform recognizes the following additional attributes
			that can be used to modify its behavior:
				
			MAX_READSIZE (expects CFNumber):  changes the maximum number of
			bytes the transform will attempt to read from the stream.  Note
			that the transform may deliver fewer bytes than this depending
			on the stream being used.
*/

/*!
	@function	SecTransformCreateReadTransformWithReadStream
	
	@abstract	Creates a read transform from a CFReadStreamRef
	
	@param inputStream	The stream that is to be opened and read from when
				the chain executes.
*/

SecTransformRef SecTransformCreateReadTransformWithReadStream(CFReadStreamRef inputStream)
API_DEPRECATED("SecTransform is no longer supported", macos(10.7, 13.0)) API_UNAVAILABLE(ios, tvos, watchos, macCatalyst);

CF_IMPLICIT_BRIDGING_DISABLED
CF_ASSUME_NONNULL_END

#ifdef __cplusplus
};
#endif

#endif

