/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 27, 2024.
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

#ifdef __cplusplus
extern "C" {
#endif

#ifndef kGenericError
	#define kGenericError		-1
#endif

extern char	*gErrorMessage;


void SetErrorMessage(const char *theErrorMessage);
void SetErrorMessageAndAppendLongInt(const char *theErrorMessage,const long theLongInt);
void SetErrorMessageAndCStrAndLongInt(const char *theErrorMessage,const char * theCStr,const long theLongInt);
void SetErrorMessageAndCStr(const char *theErrorMessage,const char * theCStr);
void AppendCStrToErrorMessage(const char *theErrorMessage);
void AppendLongIntToErrorMessage(const long theLongInt);


char *GetErrorMessage(void);
OSErr GetErrorMessageInNewHandle(Handle *inoutHandle);
OSErr GetErrorMessageInExistingHandle(Handle inoutHandle);
OSErr AppendErrorMessageToHandle(Handle inoutHandle);


#ifdef __EXCEPTIONS_ENABLED__
	void ThrowErrorMessageException(void);
#endif



//	A bunch of evil macros that would be unnecessary if I were always using C++ !

#define SetErrorMessageAndBailIfNil(theArg,theMessage)								\
{																					\
	if (theArg == nil)																\
	{																				\
		SetErrorMessage(theMessage);												\
		errCode = kGenericError;													\
		goto EXITPOINT;																\
	}																				\
}


#define SetErrorMessageAndBail(theMessage)											\
{																					\
		SetErrorMessage(theMessage);												\
		errCode = kGenericError;													\
		goto EXITPOINT;																\
}


#define SetErrorMessageAndLongIntAndBail(theMessage,theLongInt)						\
{																					\
		SetErrorMessageAndAppendLongInt(theMessage,theLongInt);						\
		errCode = kGenericError;													\
		goto EXITPOINT;																\
}


#define SetErrorMessageAndLongIntAndBailIfError(theErrCode,theMessage,theLongInt)	\
{																					\
	if (theErrCode != noErr)														\
	{																				\
		SetErrorMessageAndAppendLongInt(theMessage,theLongInt);						\
		errCode = theErrCode;														\
		goto EXITPOINT;																\
	}																				\
}


#define SetErrorMessageCStrLongIntAndBailIfError(theErrCode,theMessage,theCStr,theLongInt)	\
{																					\
	if (theErrCode != noErr)														\
	{																				\
		SetErrorMessageAndCStrAndLongInt(theMessage,theCStr,theLongInt);			\
		errCode = theErrCode;														\
		goto EXITPOINT;																\
	}																				\
}


#define SetErrorMessageAndCStrAndBail(theMessage,theCStr)							\
{																					\
	SetErrorMessageAndCStr(theMessage,theCStr);										\
	errCode = kGenericError;														\
	goto EXITPOINT;																	\
}


#define SetErrorMessageAndBailIfError(theErrCode,theMessage)						\
{																					\
	if (theErrCode != noErr)														\
	{																				\
		SetErrorMessage(theMessage);												\
		errCode = theErrCode;														\
		goto EXITPOINT;																\
	}																				\
}


#define SetErrorMessageAndLongIntAndBailIfNil(theArg,theMessage,theLongInt)			\
{																					\
	if (theArg == nil)																\
	{																				\
		SetErrorMessageAndAppendLongInt(theMessage,theLongInt);						\
		errCode = kGenericError;													\
		goto EXITPOINT;																\
	}																				\
}


#define BailIfError(theErrCode)														\
{																					\
	if ((theErrCode) != noErr)														\
	{																				\
		goto EXITPOINT;																\
	}																				\
}


#define SetErrCodeAndBail(theErrCode)												\
{																					\
	errCode = theErrCode;															\
																					\
	goto EXITPOINT;																	\
}


#define SetErrorCodeAndMessageAndBail(theErrCode,theMessage)						\
{																					\
	SetErrorMessage(theMessage);													\
	errCode = theErrCode;															\
	goto EXITPOINT;																	\
}


#define BailNow()																	\
{																					\
	errCode = kGenericError;														\
	goto EXITPOINT;																	\
}


#ifdef __cplusplus
}
#endif
