/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 9, 2025.
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
#include "DataStorageLibrary.h"
#include "CommonCode.h"


DataStorageLibrary *DataStorageLibrary::gDL;
pthread_mutex_t *DataStorageLibrary::gGlobalLock;

DataStorageLibrary::DataStorageLibrary (pthread_mutex_t *globalLock,
										CSSM_SPI_ModuleEventHandler eventHandler,
										void* CssmNotifyCallbackCtx)
	: mEventHandler (eventHandler), mCallbackContext (CssmNotifyCallbackCtx)
{
	// retain a global pointer to this library (OK because we only instantiate this object once
	gDL = this;
	gGlobalLock = globalLock;
}



DataStorageLibrary::~DataStorageLibrary ()
{
}



void DataStorageLibrary::Attach (const CSSM_GUID *ModuleGuid,
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
	// make and initialize a new AttachedInstance
	AttachedInstance* ai = MakeAttachedInstance ();
	ai->SetUpcalls (ModuleHandle, Upcalls);
	ai->Initialize (ModuleGuid, Version, SubserviceID, SubserviceType, AttachFlags,
					KeyHierarchy, CssmGuid, ModuleManagerGuid, CallerGuid);
	
	*FuncTbl = AttachedInstance::gFunctionTablePtr;
	
	// map the function to the id
	mInstanceMap[ModuleHandle] = ai;
}



void DataStorageLibrary::Detach (CSSM_MODULE_HANDLE moduleHandle)
{
	MutexLocker m (mInstanceMapMutex);
	AttachedInstance* ai = mInstanceMap[moduleHandle];
	delete ai;
}



AttachedInstance* DataStorageLibrary::HandleToInstance (CSSM_MODULE_HANDLE handle)
{
	MutexLocker _m (mInstanceMapMutex);
	
	InstanceMap::iterator m = mInstanceMap.find (handle);
	if (m == mInstanceMap.end ())
	{
		CSSMError::ThrowCSSMError(CSSMERR_DL_INVALID_DL_HANDLE);
	}

	return m->second;
}
