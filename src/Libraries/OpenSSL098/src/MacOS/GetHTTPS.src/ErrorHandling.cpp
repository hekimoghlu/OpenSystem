/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 29, 2022.
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
#include "ErrorHandling.hpp"
#include "CPStringUtils.hpp"

#ifdef __EXCEPTIONS_ENABLED__
	#include "CMyException.hpp"
#endif


static char					gErrorMessageBuffer[512];

char 						*gErrorMessage = gErrorMessageBuffer;
int							gErrorMessageMaxLength = sizeof(gErrorMessageBuffer);



void SetErrorMessage(const char *theErrorMessage)
{
	if (theErrorMessage != nil)
	{
		CopyCStrToCStr(theErrorMessage,gErrorMessage,gErrorMessageMaxLength);
	}
}


void SetErrorMessageAndAppendLongInt(const char *theErrorMessage,const long theLongInt)
{
	if (theErrorMessage != nil)
	{
		CopyCStrAndConcatLongIntToCStr(theErrorMessage,theLongInt,gErrorMessage,gErrorMessageMaxLength);
	}
}

void SetErrorMessageAndCStrAndLongInt(const char *theErrorMessage,const char * theCStr,const long theLongInt)
{
	if (theErrorMessage != nil)
	{
		CopyCStrAndInsertCStrLongIntIntoCStr(theErrorMessage,theCStr,theLongInt,gErrorMessage,gErrorMessageMaxLength);
	}

}

void SetErrorMessageAndCStr(const char *theErrorMessage,const char * theCStr)
{
	if (theErrorMessage != nil)
	{
		CopyCStrAndInsertCStrLongIntIntoCStr(theErrorMessage,theCStr,-1,gErrorMessage,gErrorMessageMaxLength);
	}
}


void AppendCStrToErrorMessage(const char *theErrorMessage)
{
	if (theErrorMessage != nil)
	{
		ConcatCStrToCStr(theErrorMessage,gErrorMessage,gErrorMessageMaxLength);
	}
}


void AppendLongIntToErrorMessage(const long theLongInt)
{
	ConcatLongIntToCStr(theLongInt,gErrorMessage,gErrorMessageMaxLength);
}



char *GetErrorMessage(void)
{
	return gErrorMessage;
}


OSErr GetErrorMessageInNewHandle(Handle *inoutHandle)
{
OSErr		errCode;


	errCode = CopyCStrToNewHandle(gErrorMessage,inoutHandle);
	
	return(errCode);
}


OSErr GetErrorMessageInExistingHandle(Handle inoutHandle)
{
OSErr		errCode;


	errCode = CopyCStrToExistingHandle(gErrorMessage,inoutHandle);
	
	return(errCode);
}



OSErr AppendErrorMessageToHandle(Handle inoutHandle)
{
OSErr		errCode;


	errCode = AppendCStrToHandle(gErrorMessage,inoutHandle,nil);
	
	return(errCode);
}


#ifdef __EXCEPTIONS_ENABLED__

void ThrowErrorMessageException(void)
{
	ThrowDescriptiveException(gErrorMessage);
}

#endif
