/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 12, 2023.
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

// $OpenLDAP$
/*
 * Copyright 2000-2011 The OpenLDAP Foundation, All Rights Reserved.
 * COPYING RESTRICTIONS APPLY, see COPYRIGHT file
 */

#ifndef LDAP_ADD_REQUEST_H
#define  LDAP_ADD_REQUEST_H

#include <LDAPRequest.h>
#include <LDAPEntry.h>

class LDAPMessageQueue;

class LDAPAddRequest : LDAPRequest {
    public:
        LDAPAddRequest(const LDAPAddRequest& req);
        LDAPAddRequest(const LDAPEntry* entry, 
                LDAPAsynConnection *connect,
                const LDAPConstraints *cons, bool isReferral=false, 
                const LDAPRequest* parent=0);
        virtual ~LDAPAddRequest();
        virtual LDAPMessageQueue* sendRequest();
        virtual LDAPRequest* followReferral(LDAPMsg* refs);
    private:
        LDAPEntry* m_entry;

};
#endif // LDAP_ADD_REQUEST_H

