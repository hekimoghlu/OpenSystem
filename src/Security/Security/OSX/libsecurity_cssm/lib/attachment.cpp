/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 28, 2025.
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
// attachment - CSSM module attachment objects
//
#include "attachment.h"
#include "module.h"
#include "manager.h"
#include "cssmcontext.h"
#include <security_cdsa_utilities/cssmbridge.h>

//
// Construct an Attachment object.
// This constructor does almost all the work: it initializes the Attachment
// object, calls the plugin's attach function, and initializes everything.
// The only job left for the subclass's constructor is to take the spiFunctionTable
// field and extract from it the plugin's dispatch table in suitable form.
//
Attachment::Attachment(Module *parent,
                       const CSSM_VERSION &version,
                       uint32 ssId,
                       CSSM_SERVICE_TYPE ssType,
                       const CSSM_API_MEMORY_FUNCS &memoryOps,
                       CSSM_ATTACH_FLAGS attachFlags,
                       CSSM_KEY_HIERARCHY keyHierarchy)
	: CssmMemoryFunctionsAllocator(memoryOps), module(*parent)
{
    // record our origins
    mVersion = version;
    mSubserviceId = ssId;
    mSubserviceType = ssType;
    mAttachFlags = attachFlags;
    mKeyHierarchy = keyHierarchy;

    // we are not (yet) attached to our plugin
    mIsActive = false;
    
    // build the upcalls table
    // (we could do this once in a static, but then we'd have to lock on it)
    upcalls.malloc_func = upcallMalloc;
    upcalls.free_func = upcallFree;
    upcalls.realloc_func = upcallRealloc;
    upcalls.calloc_func = upcallCalloc;
    upcalls.CcToHandle_func = upcallCcToHandle;
    upcalls.GetModuleInfo_func = upcallGetModuleInfo;

    // tell the module to create an attachment
    spiFunctionTable = NULL;	// preset invalid
    if (CSSM_RETURN err = module.plugin->attach(&module.myGuid(),
            &mVersion,
            mSubserviceId,
            mSubserviceType,
            mAttachFlags,
            handle(),
            mKeyHierarchy,
            &gGuidCssm,			// CSSM's Guid
            &gGuidCssm,			// module manager Guid
            &module.cssm.callerGuid(), // caller Guid
            &upcalls,
            &spiFunctionTable)) {
        // attach rejected by module
		secinfo("cssm", "attach of module %p(%s) failed",
			&module, module.name().c_str());
        CssmError::throwMe(err);
    }
    try {
        if (spiFunctionTable == NULL || spiFunctionTable->ServiceType != subserviceType())
            CssmError::throwMe(CSSMERR_CSSM_INVALID_ADDIN_FUNCTION_TABLE);
        mIsActive = true;	// now officially attached to plugin
		secinfo("cssm", "%p attached module %p(%s) (ssid %ld type %ld)",
			this, parent, parent->name().c_str(), (long)ssId, (long)ssType);
        // subclass is responsible for taking spiFunctionTable and build
        // whatever dispatch is needed
    } catch (...) {
        module.plugin->detach(handle());	// with extreme prejudice
        throw;
    }
}


//
// Detach an attachment.
// This is the polite way to detach from the plugin. It may be refused safely
// (though perhaps not meaningfully).
// THREADS: mLock is locked on entry IFF isLocked, and will be unlocked on exit.
//
void Attachment::detach(bool isLocked)
{
    StLock<Mutex> locker(*this, isLocked);	// pre-state locker
	locker.lock();	// make sure it's locked

    if (mIsActive) {
        if (!isIdle())
            CssmError::throwMe(CSSM_ERRCODE_FUNCTION_FAILED);	//@#attachment busy
        if (CSSM_RETURN error = module.plugin->detach(handle()))
			CssmError::throwMe(error);	// I'm sorry Dave, ...
		secinfo("cssm", "%p detach module %p(%s)", this,
			&module, module.name().c_str());
        mIsActive = false;
        module.detach(this);
    }
}


//
// Destroy the Attachment object
//
Attachment::~Attachment()
{
    try {
        detach(false);
    } catch (...) {
        // too bad - you're dead
    }
}


//
// Upcall relays.
// These do not lock the attachment object. The attachment can't go away
// because we incremented the busy count on entry to the plugin; and these
// fields are quite constant for the life of the Attachment.
//
void *Attachment::upcallMalloc(CSSM_HANDLE handle, size_t size)
{
    BEGIN_API_NO_METRICS
    return HandleObject::find<Attachment>(handle, CSSMERR_CSSM_INVALID_ADDIN_HANDLE).malloc(size);
    END_API1_NO_METRICS(NULL)
}

void Attachment::upcallFree(CSSM_HANDLE handle, void *mem)
{
    BEGIN_API_NO_METRICS
    return HandleObject::find<Attachment>(handle, CSSMERR_CSSM_INVALID_ADDIN_HANDLE).free(mem);
    END_API0_NO_METRICS
}

void *Attachment::upcallRealloc(CSSM_HANDLE handle, void *mem, size_t size)
{
    BEGIN_API_NO_METRICS
    return HandleObject::find<Attachment>(handle, CSSMERR_CSSM_INVALID_ADDIN_HANDLE).realloc(mem, size);
    END_API1_NO_METRICS(NULL)
}

void *Attachment::upcallCalloc(CSSM_HANDLE handle, size_t num, size_t size)
{
    BEGIN_API_NO_METRICS
    return HandleObject::find<Attachment>(handle, CSSMERR_CSSM_INVALID_ADDIN_HANDLE).calloc(size, num);
    END_API1_NO_METRICS(NULL)
}

CSSM_RETURN Attachment::upcallCcToHandle(CSSM_CC_HANDLE handle,
                                         CSSM_MODULE_HANDLE *modHandle)
{
    BEGIN_API_NO_METRICS
    Required(modHandle) = HandleObject::find<HandleContext>((CSSM_HANDLE)handle, CSSMERR_CSSM_INVALID_ADDIN_HANDLE).attachment.handle();
    END_API_NO_METRICS(CSP)
}

CSSM_RETURN Attachment::upcallGetModuleInfo(CSSM_MODULE_HANDLE handle,
                                            CSSM_GUID_PTR guid,
                                            CSSM_VERSION_PTR version,
                                            uint32 *subserviceId,
                                            CSSM_SERVICE_TYPE *subserviceType,
                                            CSSM_ATTACH_FLAGS *attachFlags,
                                            CSSM_KEY_HIERARCHY *keyHierarchy,
                                            CSSM_API_MEMORY_FUNCS_PTR memoryOps,
                                            CSSM_FUNC_NAME_ADDR_PTR FunctionTable,
                                            uint32 NumFunctions)
{
    BEGIN_API_NO_METRICS
    Attachment &attachment = HandleObject::find<Attachment>(handle, CSSMERR_CSSM_INVALID_ADDIN_HANDLE);
    Required(guid) = attachment.myGuid();
    Required(version) = attachment.mVersion;
    Required(subserviceId) = attachment.mSubserviceId;
    Required(subserviceType) = attachment.mSubserviceType;
    Required(attachFlags) = attachment.mAttachFlags;
    Required(keyHierarchy) = attachment.mKeyHierarchy;
    Required(memoryOps) = attachment;
    if (FunctionTable)
        attachment.resolveSymbols(FunctionTable, NumFunctions);
    END_API_NO_METRICS(CSSM)
}
