/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 20, 2025.
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

#ifndef LDAP_SEARCH_REQUEST_H
#define LDAP_SEARCH_REQUEST_H

#include <queue>
#include <LDAPRequest.h>

class LDAPSearchReference;
class LDAPReferral;
class LDAPUrl;

class LDAPSearchRequest : public LDAPRequest{ 

    public :
        LDAPSearchRequest(const LDAPSearchRequest& req);

        LDAPSearchRequest(const std::string& base, int scope, const std::string& filter,
                          const StringList& attrs, bool attrsOnly, 
                          LDAPAsynConnection *connect,
                          const LDAPConstraints* cons, bool isReferral=false,
                          const LDAPRequest* parent=0);
        virtual ~LDAPSearchRequest();        
        virtual LDAPMessageQueue* sendRequest();
        virtual LDAPRequest* followReferral(LDAPMsg* ref);
        virtual bool equals(const LDAPRequest* req) const;
    
    private :
        std::string m_base;
        int m_scope;
        std::string m_filter;
        StringList m_attrs;
        bool m_attrsOnly;

        //no default constructor
        LDAPSearchRequest(){};
};

#endif //LDAP_SEARCH_REQUEST_H
