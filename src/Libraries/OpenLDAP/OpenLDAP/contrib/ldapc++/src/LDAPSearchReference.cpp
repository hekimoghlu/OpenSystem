/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 12, 2022.
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


#include <iostream>

#include "debug.h"
#include "LDAPSearchReference.h"
#include "LDAPException.h"
#include "LDAPRequest.h"
#include "LDAPUrl.h"

using namespace std;

LDAPSearchReference::LDAPSearchReference(const LDAPRequest *req,
        LDAPMessage *msg) : LDAPMsg(msg){
    DEBUG(LDAP_DEBUG_CONSTRUCT,
            "LDAPSearchReference::LDAPSearchReference()" << endl;)    
    char **ref=0;
    LDAPControl** srvctrls=0;
    const LDAPAsynConnection* con=req->getConnection();
    int err = ldap_parse_reference(con->getSessionHandle(), msg, &ref, 
            &srvctrls,0);
    if (err != LDAP_SUCCESS){
        ber_memvfree((void**) ref);
        ldap_controls_free(srvctrls);
        throw LDAPException(err);
    }else{
        m_urlList=LDAPUrlList(ref);
        ber_memvfree((void**) ref);
        if (srvctrls){
            m_srvControls = LDAPControlSet(srvctrls);
            m_hasControls = true;
            ldap_controls_free(srvctrls);
        }else{
            m_hasControls = false;
        }
    }
}

LDAPSearchReference::~LDAPSearchReference(){
    DEBUG(LDAP_DEBUG_DESTROY,"LDAPSearchReference::~LDAPSearchReference()"
            << endl);
}

const LDAPUrlList& LDAPSearchReference::getUrls() const{
    DEBUG(LDAP_DEBUG_TRACE,"LDAPSearchReference::getUrls()" << endl);
    return m_urlList;
}

