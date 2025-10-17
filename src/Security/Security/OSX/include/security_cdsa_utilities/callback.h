/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 15, 2021.
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
#ifndef _H_CALLBACK
#define _H_CALLBACK

#include <Security/cssm.h>
#include <security_utilities/threading.h>
#include <security_cdsa_utilities/cssmpods.h>
#include <map>

namespace Security
{

//
// A single module-specific callback as requested by the user.
//
class ModuleCallback {
public:
	ModuleCallback() : mCallback(0), mContext(0) { }
    ModuleCallback(CSSM_API_ModuleEventHandler callback, void *context)
    : mCallback(callback), mContext(context) { }

    void operator () (CSSM_MODULE_EVENT event,
                      const Guid &guid, uint32 subId,
                      CSSM_SERVICE_TYPE serviceType) const;

    operator bool () const { return mCallback || mContext; }
    bool operator ! () const { return !bool(*this); }

    bool operator == (const ModuleCallback &cb) const
    { return mCallback == cb.mCallback && mContext == cb.mContext; }
    bool operator < (const ModuleCallback &cb) const
    { return (uintptr_t)mCallback < (uintptr_t)cb.mCallback
        || (mCallback == cb.mCallback && mContext < cb.mContext); }

private:
    CSSM_API_ModuleEventHandler mCallback;
    void *mContext;
};


//
// A set of callbacks that can be invoked automatically in a thread-safe manner.
// THREADS: The set itself is not interlocked by the ModuleCallbackSet class; you
// are responsible for ensuring single access to the set object. The class ensures
// that any threads it spawns to execute the callbacks will not step on each other
// or on you, and that you will not be able to erase() a callback while it has
// activity scheduled against it. This also applies to the invocation method
// (operator ()) - you must lock against multiple accesses to it until it returns.
//
class ModuleCallbackSet {
public:
    unsigned int size() const { return (int)callbacks.size(); }
    void insert(const ModuleCallback &newCallback);
    void erase(const ModuleCallback &oldCallback);

    void operator () (CSSM_MODULE_EVENT event,
                      const Guid &guid, uint32 subId,
                      CSSM_SERVICE_TYPE serviceType) const;

private:
    // note mutex *: we don't want to rely on copy-ability of Mutex objects
    typedef multimap<ModuleCallback, CountingMutex *> CallbackMap;
    mutable CallbackMap callbacks;
};

} // end namespace Security


#endif //_H_CALLBACK
