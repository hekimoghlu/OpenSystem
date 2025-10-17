/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 16, 2024.
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
/*!
 @header EncryptTransformUtilities
 This file contains utilities used by the SecEncryptTransform and the SecDecryptTransform
 
 */
#if !defined(___ENCRYPT_TRANSFORM_UTILITIES__)
#define ___ENCRYPT_TRANSFORM_UTILITIES__ 1

#include <CoreFoundation/CoreFoundation.h> 
#include <Security/cssmapi.h>
#include <Security/cssmapple.h>
#include <Security/cssmtype.h>

#ifdef __cplusplus
extern "C" {
#endif
	
	/*!
	 @function 			ConvertPaddingStringToEnum
	 @abstract			Given a string that represents a padding return the
	 CSSM_PADDING value
	 @param paddingStr	A CFStringRef that represents a padding string
	 @result				The corresponding CSSM_PADDING value or -1 if the
	 padding value could not be found
	 */
	uint32	ConvertPaddingStringToEnum(CFStringRef paddingStr);
	
	/*!
	 @function 			ConvertPaddingEnumToString
	 @abstract			Given a CSSM_PADDING value return the corresponding 
	 CFString representation.
	 @param paddingEnum	A CSSM_PADDING value.
	 @result				The corresponding CFStringRef or NULL if the the 
	 CSSM_PADDING value could not be found
	 */
	CFStringRef ConvertPaddingEnumToString(CSSM_PADDING paddingEnum);
	
	
	/*!
	 @function 			ConvertEncryptModeStringToEnum
	 @abstract			Given a string that represents an encryption mode return the
	 CSSM_ENCRYPT_MODE value
	 @param modeStr	A CFStringRef that represents an encryption mode
     @param hasPadding Specify if the mode should pad
		 @result				The corresponding CSSM_ENCRYPT_MODE value or -1 if the
	 encryptio mode value could not be found
	 */
	uint32	ConvertEncryptModeStringToEnum(CFStringRef modeStr, Boolean hasPadding);
	
	/*!
	 @function 			ConvertPaddingEnumToString
	 @abstract			Given a CSSM_ENCRYPT_MODE value return the corresponding 
	 CFString representation.
	 @param paddingEnum	A CSSM_ENCRYPT_MODE value.
	 @result				The corresponding CFStringRef or NULL if the the 
	 CSSM_ENCRYPT_MODE value could not be found
	 */
	CFStringRef ConvertEncryptModeEnumToString(CSSM_ENCRYPT_MODE paddingEnum);
	
#ifdef __cplusplus
}
#endif

#endif /* !___ENCRYPT_TRANSFORM_UTILITIES__ */
