/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 22, 2021.
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
#include <Security/Security.h>
#include "DataStorageLibrary.h"



/*
	stubs for CSSM SPI's -- these simply call the next level
*/



extern "C" CSSM_RETURN CSSM_SPI_ModuleUnload (const CSSM_GUID *CssmGuid,
											  const CSSM_GUID *ModuleGuid,
											  CSSM_SPI_ModuleEventHandler CssmNotifyCallback,
											  void* CssmNotifyCallbackCtx)
{
	try
	{
		StaticMutex _mutex (*DataStorageLibrary::gGlobalLock);
		MutexLocker _lock (_mutex);
		
		if (DataStorageLibrary::gDL != NULL)
		{
			// delete our instance
			delete DataStorageLibrary::gDL;
			DataStorageLibrary::gDL = NULL;
		}
		return 0;
	}
	catch (...)
	{
		return CSSMERR_CSSM_INTERNAL_ERROR;
	}
}



extern "C" CSSM_RETURN CSSM_SPI_ModuleAttach (const CSSM_GUID *ModuleGuid,
											  const CSSM_VERSION *Version,
											  uint32 SubserviceID,
											  CSSM_SERVICE_TYPE SubserviceType,
											  CSSM_ATTACH_FLAGS AttachFlags,
											  CSSM_MODULE_HANDLE ModuleHandle,
											  CSSM_KEY_HIERARCHY KeyHierarchy,
											  const CSSM_GUID *CssmGuid,
											  const CSSM_GUID *ModuleManagerGuid,
											  const CSSM_GUID *CallerGuid,
											  const CSSM_UPCALLS *Upcalls,
											  CSSM_MODULE_FUNCS_PTR *FuncTbl)
{
	try
	{
		if (DataStorageLibrary::gDL == NULL)
		{
			return CSSMERR_CSSM_MODULE_NOT_LOADED;
		}
		
		DataStorageLibrary::gDL->Attach (ModuleGuid, Version, SubserviceID, SubserviceType, AttachFlags, ModuleHandle, KeyHierarchy,
										 CssmGuid, ModuleManagerGuid, CallerGuid, Upcalls, FuncTbl);
		return 0;
	}
	catch (...)
	{
		return CSSMERR_CSSM_INTERNAL_ERROR;
	}
}



extern "C" CSSM_RETURN CSSM_SPI_ModuleDetach (CSSM_MODULE_HANDLE ModuleHandle)
{
	try
	{
		if (DataStorageLibrary::gDL == NULL)
		{
			return CSSMERR_CSSM_MODULE_NOT_LOADED;
		}
		
		DataStorageLibrary::gDL->Detach (ModuleHandle);
		return 0;
	}
	catch (...)
	{
		return CSSMERR_CSSM_INTERNAL_ERROR;
	}
}
