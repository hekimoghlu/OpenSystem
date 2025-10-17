/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 29, 2023.
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

#ifndef LDAP_MODIFY_REQUEST_H
#define LDAP_MODIFY_REQUEST_H

#include <LDAPRequest.h>

class LDAPMessageQueue;

class LDAPModifyRequest : LDAPRequest {
    private :
        std::string m_dn;
        LDAPModList *m_modList;

    public:
        LDAPModifyRequest(const LDAPModifyRequest& mod);
        LDAPModifyRequest(const std::string& dn, const LDAPModList *modList,
                LDAPAsynConnection *connect, const LDAPConstraints *cons,
                bool isReferral=false, const LDAPRequest* req=0);
        virtual ~LDAPModifyRequest();
        virtual LDAPMessageQueue* sendRequest();
        virtual LDAPRequest* followReferral(LDAPMsg* refs);
};

#endif // LDAP_MODIFY_REQUEST_H

