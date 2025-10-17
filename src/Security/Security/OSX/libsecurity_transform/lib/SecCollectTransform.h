/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 14, 2022.
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

#ifndef __SECCOLLECTTRANSFORM_H__
#define __SECCOLLECTTRANSFORM_H__

/*
 * Copyright (c) 2010-2011 Apple Inc. All Rights Reserved.
 * 
 * @APPLE_LICENSE_HEADER_START@
 * 
 * This file contains Original Code and/or Modifications of Original Code
 * as defined in and that are subject to the Apple Public Source License
 * Version 2.0 (the 'License'). You may not use this file except in
 * compliance with the License. Please obtain a copy of the License at
 * http://www.opensource.apple.com/apsl/ and read it before using this
 * file.
 * 
 * The Original Code and all software distributed under the License are
 * distributed on an 'AS IS' basis, WITHOUT WARRANTY OF ANY KIND, EITHER
 * EXPRESS OR IMPLIED, AND APPLE HEREBY DISCLAIMS ALL SUCH WARRANTIES,
 * INCLUDING WITHOUT LIMITATION, ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE, QUIET ENJOYMENT OR NON-INFRINGEMENT.
 * Please see the License for the specific language governing rights and
 * limitations under the License.
 * 
 * @APPLE_LICENSE_HEADER_END@
 */

#include "SecTransform.h"

#ifdef __cplusplus
extern "C" {
#endif


/*!
	 @function 			SecCreateCollectTransform
	 @abstract			Creates a  collection object.
	
	
	 @param error		A pointer to a CFErrorRef.  This pointer will be set
	 					if an error occurred.  This value may be NULL if you
	 					do not want an error returned.
	 @result			A pointer to a SecTransformRef object.  This object must
	 					be released with CFRelease when you are done with
						it.  This function will return NULL if an error
						occurred.
						
	 @discussion		This function creates a transform will collect all
						of the data given to it and output a single data 
						item
*/


SecTransformRef SecCreateCollectTransform(CFErrorRef* error)
	__OSX_AVAILABLE_STARTING(__MAC_10_7,__IPHONE_NA);

#ifdef __cplusplus
}
#endif

#endif // __SECCOLLECTTRANSFORM_H__
