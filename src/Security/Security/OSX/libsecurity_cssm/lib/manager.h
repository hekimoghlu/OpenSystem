/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 5, 2024.
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
// manager - CSSM manager/supervisor objects.
//
#ifndef _H_MANAGER
#define _H_MANAGER

#include "modloader.h"
#include <security_cdsa_utilities/callback.h>
#include "cssmmds.h"
#include "attachfactory.h"


//
// The CssmManager class embodies one instance of CSSM. It can interact with multiple
// callers in multiple threads.
// As far as CssmManager is concerned, it doesn't mind for multiple instances of it to
// exist. Such instances are strictly separated; they do not share anything (module info,
// attachments, callbacks, etc.) and live their lives in splendid isolation. Of course,
// other subsystems they deal with (e.g. the ModuleLoader) may multiplex them, but such
// components should take pains not to "leak" information from one CssmManager instance
// to another.
//
class CssmManager : public AttachmentFactory {
    NOCOPY(CssmManager)
    static const CSSM_GUID theGuidForCssmItself;
public:
    CssmManager();
    virtual ~CssmManager();
    
    void initialize (const CSSM_VERSION &version,
                     CSSM_PRIVILEGE_SCOPE scope,
                     const Guid &callerGuid,
                     CSSM_KEY_HIERARCHY keyHierarchy,
                     CSSM_PVC_MODE &pvcPolicy);
    bool terminate();

    void loadModule (const Guid &guid,
                     CSSM_KEY_HIERARCHY keyHierarchy,
                     const ModuleCallback &callback);
    void unloadModule (const Guid &guid,
                       const ModuleCallback &callback);

    void introduce(const Guid &guid, CSSM_KEY_HIERARCHY keyHierarchy);
    void unIntroduce(const Guid &guid);

    //
    // Polite inquiries
    //

    // these values are constant (after init time) and need no locking
    const Guid &callerGuid() const { return mCallerGuid; }
    CSSM_PRIVILEGE_SCOPE privilegeScope() const { return mPrivilegeScope; }
    CSSM_KEY_HIERARCHY keyHierarchy() const { return mKeyHierarchy; }
    CSSM_PVC_MODE pvcMode() const { return mPvcPolicy; }

    //@@@ for these two, consider locking (as of the C shims AND the transition layer use)
    const CSSM_PRIVILEGE &getPrivilege() const { return mPrivilege; }
    void setPrivilege(const CSSM_PRIVILEGE &priv) { mPrivilege = priv; }

public:
    Module *getModule(const Guid &guid);

private:
    typedef map<Guid, Module *> ModuleMap;
    ModuleMap moduleMap;

    Mutex mLock;					// object lock
    unsigned int initCount;			// number of times successfully initialized
    
private:
    ModuleLoader loader;			// our ticket to module land

private:
    // state acquired from initialize (instance constants - not guarded)
    CSSM_PRIVILEGE_SCOPE mPrivilegeScope;
    CSSM_KEY_HIERARCHY mKeyHierarchy;
    CSSM_PVC_MODE mPvcPolicy;
    Guid mCallerGuid;

    // persistent state of the CSSM (guarded by module lock)
    CSSM_PRIVILEGE mPrivilege;		// established privileges

private:
    void checkVersion(const CSSM_VERSION &version);
};

#ifdef _CPP_MANAGER
# pragma export off
#endif

#endif //_H_MANAGER
