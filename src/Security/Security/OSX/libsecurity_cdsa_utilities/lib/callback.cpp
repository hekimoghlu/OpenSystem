/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 7, 2023.
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
// Encapsulate the callback mechanism of CSSM.
//
#include <security_cdsa_utilities/callback.h>


//
// Invoke a callback
//
void ModuleCallback::operator () (CSSM_MODULE_EVENT event,
                                  const Guid &guid, uint32 subId,
                                  CSSM_SERVICE_TYPE serviceType) const
{
    try
    {
        if (mCallback)
            if (CSSM_RETURN err = mCallback(&guid, mContext, subId, serviceType, event))
                CssmError::throwMe(err);
    }
    catch (...)
    {
    }
}


//
// Manage Callback sets.
// THREADS: Caller is ensuring single-thread access on these calls.
//
void ModuleCallbackSet::insert(const ModuleCallback &newCallback)
{
    callbacks.insert(CallbackMap::value_type(newCallback, new CountingMutex));
}

void ModuleCallbackSet::erase(const ModuleCallback &oldCallback)
{
    CallbackMap::iterator it = callbacks.find(oldCallback);
    if (it == callbacks.end())	// not registered; fail
        CssmError::throwMe(CSSMERR_CSSM_INVALID_ADDIN_HANDLE);
    CountingMutex *counter = it->second;
    {
        StLock<Mutex> _(*counter);
        if (!counter->isIdle()) // callbacks are scheduled against this
            CssmError::throwMe(CSSM_ERRCODE_FUNCTION_FAILED);	// @#module is busy
    }
    // counter is zero (idle), and we hold the entry lock (via our caller)
    delete counter;
    callbacks.erase(it);
}


//
// Invoke an entire callback set.
// THREADS: Caller is ensuring  single-thread access on these calls.
// Note that the callbacks are background thread, and so is the ->exit() call
//
// NOTE WELL, if this is called, you can't safely call add/remove

void ModuleCallbackSet::operator () (CSSM_MODULE_EVENT event,
                                     const Guid &guid, uint32 subId,
                                     CSSM_SERVICE_TYPE serviceType) const
{
    if (callbacks.empty())	// nothing to do; quick exit
        return;
    
    for (auto it = callbacks.begin(); it != callbacks.end(); it++) {
        it->second->enter();
    }

    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        for (auto it = callbacks.begin(); it != callbacks.end(); it++) {
            it->first(event, guid, subId, serviceType);
            it->second->exit();
        }
    });
}
