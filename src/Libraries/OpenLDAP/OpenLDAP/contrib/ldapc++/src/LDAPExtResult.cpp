/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 21, 2022.
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

#include "debug.h"
#include <lber.h>
#include "LDAPRequest.h"
#include "LDAPException.h"

#include "LDAPResult.h"
#include "LDAPExtResult.h"

using namespace std;

LDAPExtResult::LDAPExtResult(const LDAPRequest* req, LDAPMessage* msg) :
        LDAPResult(req, msg){
    DEBUG(LDAP_DEBUG_CONSTRUCT,"LDAPExtResult::LDAPExtResult()" << endl);
    char* oid = 0;
    BerValue* data = 0;
    LDAP* lc = req->getConnection()->getSessionHandle();
    int err=ldap_parse_extended_result(lc, msg, &oid, &data, 0);
    if(err != LDAP_SUCCESS){
        ber_bvfree(data);
        ldap_memfree(oid);
        throw LDAPException(err);
    }else{
        m_oid=string(oid);
        ldap_memfree(oid);
        if(data){
            m_data=string(data->bv_val, data->bv_len);
            ber_bvfree(data);
        }
    }
}

LDAPExtResult::~LDAPExtResult(){
    DEBUG(LDAP_DEBUG_DESTROY,"LDAPExtResult::~LDAPExtResult()" << endl);
}

const string& LDAPExtResult::getResponseOid() const{
    return m_oid;
}

const string& LDAPExtResult::getResponse() const{
    return m_data;
}

