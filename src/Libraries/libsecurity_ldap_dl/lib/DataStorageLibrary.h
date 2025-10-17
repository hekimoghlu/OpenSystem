/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 12, 2022.
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
#ifndef __DATA_STORAGE_LIBRARY_H__
#define __DATA_STORAGE_LIBRARY_H__



#include <Security/Security.h>
#include "AttachedInstance.h"
#include "Mutex.h"

#include <map>


typedef std::map<CSSM_MODULE_HANDLE, AttachedInstance*> InstanceMap;

// a class which creates, deallocates, and finds attached instances

class DataStorageLibrary
{
protected:

	CSSM_SPI_ModuleEventHandler mEventHandler;
	void* mCallbackContext;
	InstanceMap mInstanceMap;
	DynamicMutex mInstanceMapMutex;

public:
	static DataStorageLibrary* gDL;
	static pthread_mutex_t *gGlobalLock;

	DataStorageLibrary (pthread_mutex_t *globalLock, CSSM_SPI_ModuleEventHandler eventHandler, void* CssmNotifyCallbackCtx);
	virtual ~DataStorageLibrary ();

	void Attach (const CSSM_GUID *ModuleGuid,
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
			     CSSM_MODULE_FUNCS_PTR *FuncTbl);
	
	void Detach (CSSM_MODULE_HANDLE moduleHandle);

	virtual AttachedInstance* MakeAttachedInstance () = 0;

	AttachedInstance* HandleToInstance (CSSM_MODULE_HANDLE handle);
};



#endif
