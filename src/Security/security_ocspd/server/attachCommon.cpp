/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 6, 2024.
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
/*
 * attachCommon.cpp - attach/detach to/from arbitrary module
 */
 
#include "attachCommon.h"
#include <Security/Security.h>

/* SPI; the framework actually contains a static lib we link against */
#include <security_cdsa_utils/cuCdsaUtils.h>

static CSSM_VERSION vers = {2, 0};
static const CSSM_GUID dummyGuid = { 0xFADE, 0, 0, { 1,2,3,4,5,6,7,0 }};

static CSSM_API_MEMORY_FUNCS memFuncs = {
	cuAppMalloc,
	cuAppFree,
	cuAppRealloc,
 	cuAppCalloc,
 	NULL
};

/* load & attach; returns 0 on error */
CSSM_HANDLE attachCommon(
	const CSSM_GUID *guid,
	uint32 subserviceFlags)		// CSSM_SERVICE_TP, etc.
{
	CSSM_HANDLE hand;
	CSSM_RETURN crtn;
	
	if(cuCssmStartup() == CSSM_FALSE) {
		return 0;
	}
	crtn = CSSM_ModuleLoad(guid,
		CSSM_KEY_HIERARCHY_NONE,
		NULL,			// eventHandler
		NULL);			// AppNotifyCallbackCtx
	if(crtn) {
		cssmPerror("CSSM_ModuleLoad()", crtn);
		return 0;
	}
	crtn = CSSM_ModuleAttach (guid,
		&vers,
		&memFuncs,				// memFuncs
		0,						// SubserviceID
		subserviceFlags,		// SubserviceFlags
		0,						// AttachFlags
		CSSM_KEY_HIERARCHY_NONE,
		NULL,					// FunctionTable
		0,						// NumFuncTable
		NULL,					// reserved
		&hand);
	if(crtn) {
		cssmPerror("CSSM_ModuleAttach()", crtn);
		return 0;
	}
	else {
		return hand;
	}
}

/* detach & unload */
void detachCommon(
	const CSSM_GUID *guid,
	CSSM_HANDLE hand)
{
	CSSM_RETURN crtn = CSSM_ModuleDetach(hand);
	if(crtn) {
		return;
	}
	CSSM_ModuleUnload(guid, NULL, NULL);
}


