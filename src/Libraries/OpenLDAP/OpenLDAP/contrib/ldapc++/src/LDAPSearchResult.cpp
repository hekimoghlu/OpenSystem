/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 29, 2025.
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
#include "LDAPSearchResult.h"
#include "LDAPRequest.h"

using namespace std;

LDAPSearchResult::LDAPSearchResult(const LDAPRequest *req,
        LDAPMessage *msg) : LDAPMsg(msg){
	DEBUG(LDAP_DEBUG_CONSTRUCT,
		"LDAPSearchResult::LDAPSearchResult()" << endl);
    entry = new LDAPEntry(req->getConnection(), msg);
    //retrieve the controls here
    LDAPControl** srvctrls=0;
    int err = ldap_get_entry_controls(req->getConnection()->getSessionHandle(),
            msg,&srvctrls);
    if(err != LDAP_SUCCESS){
        ldap_controls_free(srvctrls);
    }else{
        if (srvctrls){
            m_srvControls = LDAPControlSet(srvctrls);
            m_hasControls = true;
            ldap_controls_free(srvctrls);
        }else{
            m_hasControls = false;
        }
    }
}

LDAPSearchResult::LDAPSearchResult(const LDAPSearchResult& res) :
        LDAPMsg(res){
    entry = new LDAPEntry(*(res.entry));
}

LDAPSearchResult::~LDAPSearchResult(){
	DEBUG(LDAP_DEBUG_DESTROY,"LDAPSearchResult::~LDAPSearchResult()" << endl);
	delete entry;
}

const LDAPEntry* LDAPSearchResult::getEntry() const{
	DEBUG(LDAP_DEBUG_TRACE,"LDAPSearchResult::getEntry()" << endl);
	return entry;
}

