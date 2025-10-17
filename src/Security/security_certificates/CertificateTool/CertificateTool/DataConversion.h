/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 5, 2023.
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

//
//  DataConversion.h
//  CertificateTool
//
//  Copyright (c) 2013-2015 Apple Inc. All Rights Reserved.
//

#import <Foundation/Foundation.h>

/*! =========================================================================
	@class NSData

	@abstract	Extend the NSData object to convert to a hex string
	========================================================================= */
@interface NSData (DataConversion)

/*! -------------------------------------------------------------------------
	@method 	toHexString

	@result		returns a NSString object with the hex characters that 
				represent the value of the NSData object.
	------------------------------------------------------------------------- */
- (NSString *)toHexString;

@end


/*! =========================================================================
	@class NSString

	@abstract	Extend the NSString object to convert hex string into a 
				binary value in a NSData object.
	========================================================================= */
@interface NSString (DataConversion)

/*! -------------------------------------------------------------------------
	@method 	hextStringToData

	@result		Convert a NSString that contains a set of hex characters 
				into a binary value.  If the conversion cannot be done then
				nil will be returned.
	------------------------------------------------------------------------- */
- (NSData *)hexStringToData;

@end


