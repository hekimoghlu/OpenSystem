/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 29, 2024.
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
 Provides utilies for the SecEncryptTransform file.
 
 */

#include "EncryptTransformUtilities.h"
#include "SecEncryptTransform.h"

/* --------------------------------------------------------------------------
 method: 		ConvertPaddingStringToEnum
 description: 	Get the CSSM_PADDING value from a CFString
 -------------------------------------------------------------------------- */
uint32	ConvertPaddingStringToEnum(CFStringRef paddingStr)
{
	uint32 result = -1;		// Guilty until proven
	if (NULL == paddingStr)
	{
		return result;
	}
	
	if (CFEqual(paddingStr, kSecPaddingNoneKey))
	{
		result = CSSM_PADDING_NONE; 
	}
	else if (CFEqual(paddingStr, kSecPaddingPKCS1Key))
	{
		result = CSSM_PADDING_PKCS7; //CSSM_PADDING_PKCS1 ois not supported
	}
	else if (CFEqual(paddingStr, kSecPaddingPKCS5Key))
	{
		result = CSSM_PADDING_PKCS5;
	}
	else if (CFEqual(paddingStr, kSecPaddingPKCS7Key))
	{
		result = CSSM_PADDING_PKCS7;
	}
	return result;	
}


/* --------------------------------------------------------------------------
 method: 		ConvertPaddingEnumToString
 description: 	Get the corresponding CFStringRef for a CSSM_PADDING value
 -------------------------------------------------------------------------- */
CFStringRef ConvertPaddingEnumToString(CSSM_PADDING paddingEnum)
{
	CFStringRef result = NULL;
	switch (paddingEnum)
	{
		case CSSM_PADDING_NONE:
			result = kSecPaddingNoneKey;
			break;
			
		case CSSM_PADDING_PKCS5:
			result = kSecPaddingPKCS5Key;
			break;
			
		case CSSM_PADDING_PKCS7:
			result = kSecPaddingPKCS7Key;
			break;
			
		case CSSM_PADDING_PKCS1:
			result = kSecPaddingPKCS1Key;
			break;
			
		default:
			result = NULL;
			break;
	}
	
	return result;
}



/* --------------------------------------------------------------------------
 method: 		ConvertEncryptModeStringToEnum
 description: 	Given a string that represents an encryption mode return the
 CSSM_ENCRYPT_MODE value
 -------------------------------------------------------------------------- */
uint32	ConvertEncryptModeStringToEnum(CFStringRef modeStr, Boolean hasPadding)
{
	uint32 result = -1;	// Guilty until proven
	
	if (NULL == modeStr)
	{
		return result;
	}
	
	if (CFEqual(modeStr, kSecModeNoneKey))
	{
		result = (hasPadding) ? CSSM_ALGMODE_ECBPad : CSSM_ALGMODE_ECB;
	}
	else if (CFEqual(modeStr, kSecModeECBKey))
	{
		result = (hasPadding) ? CSSM_ALGMODE_ECBPad : CSSM_ALGMODE_ECB;
	}
	else if (CFEqual(modeStr, kSecModeCBCKey))
	{
        result = (hasPadding) ? CSSM_ALGMODE_CBCPadIV8 : CSSM_ALGMODE_CBC;
	}
	else if (CFEqual(modeStr, kSecModeCFBKey))
	{
		result = (hasPadding) ? CSSM_ALGMODE_CFBPadIV8 : CSSM_ALGMODE_CFB;
	}
	else if (CFEqual(modeStr, kSecModeOFBKey))
	{
		result = (hasPadding) ? CSSM_ALGMODE_OFBPadIV8 : CSSM_ALGMODE_OFB;
	}
    
	return result;	
}

/* --------------------------------------------------------------------------
 method: 		ConvertEncryptModeEnumToString
 description: 	Given a CSSM_ENCRYPT_MODE value return the corresponding 
 CFString representation.
 -------------------------------------------------------------------------- */
CFStringRef ConvertEncryptModeEnumToString(CSSM_ENCRYPT_MODE paddingEnum)
{
	CFStringRef result = NULL;
	
	switch (paddingEnum)
	{
		case CSSM_ALGMODE_NONE:
		default:
			result = kSecModeNoneKey;
			break;	
			
		case CSSM_ALGMODE_ECB:
		case CSSM_ALGMODE_ECBPad:
		case CSSM_ALGMODE_ECB64:
		case CSSM_ALGMODE_ECB128:
		case CSSM_ALGMODE_ECB96:
			result = kSecModeECBKey;
			break;
			
		case CSSM_ALGMODE_CBC:
		case CSSM_ALGMODE_CBC_IV8:
		case CSSM_ALGMODE_CBCPadIV8:
		case CSSM_ALGMODE_CBC64:
		case CSSM_ALGMODE_CBC128:
			result = kSecModeCBCKey;
			break;
			
		case CSSM_ALGMODE_CFB:
		case CSSM_ALGMODE_CFB_IV8:
		case CSSM_ALGMODE_CFBPadIV8:
		case CSSM_ALGMODE_CFB32:
		case CSSM_ALGMODE_CFB16:
		case CSSM_ALGMODE_CFB8:
			result = kSecModeCFBKey;
			break;
			
		case CSSM_ALGMODE_OFB:
		case CSSM_ALGMODE_OFB_IV8:
		case CSSM_ALGMODE_OFBPadIV8:
		case CSSM_ALGMODE_OFB64:
			result = kSecModeOFBKey;
			break;		
	}
	
	return result;	
}

