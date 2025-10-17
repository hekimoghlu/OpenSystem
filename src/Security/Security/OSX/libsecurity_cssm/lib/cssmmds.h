/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 30, 2023.
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
// cssmmds - MDS interface for CSSM & friends
//
#ifndef _H_CSSMMDS
#define _H_CSSMMDS

#include "cssmint.h"
#include <security_utilities/globalizer.h>
#include <security_cdsa_utilities/cssmpods.h>
#include <security_cdsa_client/mds_standard.h>


class MdsComponent {
public:
    MdsComponent(const Guid &guid);
    virtual ~MdsComponent();

    const Guid &myGuid() const { return mMyGuid; }
    
    CSSM_SERVICE_MASK services() const { return mCommon->serviceMask(); }
    bool supportsService(CSSM_SERVICE_TYPE t) const { return t & services(); }
    bool isThreadSafe() const { return !mCommon->singleThreaded(); }
    string path() const { return mCommon->path(); }
	string name() const { return mCommon->moduleName(); }
	string description() const { return mCommon->description(); }

private:
    const Guid mMyGuid;					// GUID of the component
	RefPointer<MDSClient::Common> mCommon; // MDS common record for this module
};


#endif //_H_CSSMMDS
