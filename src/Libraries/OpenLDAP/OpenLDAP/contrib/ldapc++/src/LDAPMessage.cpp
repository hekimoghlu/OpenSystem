/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 19, 2023.
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


#include "LDAPMessage.h"

#include "LDAPResult.h"
#include "LDAPExtResult.h"
#include "LDAPSaslBindResult.h"
#include "LDAPRequest.h"
#include "LDAPSearchResult.h"
#include "LDAPSearchReference.h"
#include "debug.h"
#include <iostream>

using namespace std;

LDAPMsg::LDAPMsg(LDAPMessage *msg){
    DEBUG(LDAP_DEBUG_CONSTRUCT,"LDAPMsg::LDAPMsg()" << endl);
    msgType=ldap_msgtype(msg);
    m_hasControls=false;
}

LDAPMsg::LDAPMsg(int type, int id=0){
    DEBUG(LDAP_DEBUG_CONSTRUCT,"LDAPMsg::LDAPMsg()" << endl);
    msgType = type;
    msgID = id;
    m_hasControls=false;
}

LDAPMsg* LDAPMsg::create(const LDAPRequest *req, LDAPMessage *msg){
    DEBUG(LDAP_DEBUG_TRACE,"LDAPMsg::create()" << endl);
    switch(ldap_msgtype(msg)){
        case SEARCH_ENTRY :
            return new LDAPSearchResult(req,msg);
        break;
        case SEARCH_REFERENCE :
            return new LDAPSearchReference(req, msg);
        break;
        case EXTENDED_RESPONSE :
            return new LDAPExtResult(req,msg);
        break;
        case BIND_RESPONSE :
            return new LDAPSaslBindResult(req,msg);
        default :
            return new LDAPResult(req, msg);
    }
    return 0;
}


int LDAPMsg::getMessageType(){
    DEBUG(LDAP_DEBUG_TRACE,"LDAPMsg::getMessageType()" << endl);
    return msgType;
}

int LDAPMsg::getMsgID(){
    DEBUG(LDAP_DEBUG_TRACE,"LDAPMsg::getMsgID()" << endl);
    return msgID;
}

bool LDAPMsg::hasControls() const{
    return m_hasControls;
}

const LDAPControlSet& LDAPMsg::getSrvControls() const {
    return m_srvControls;
}

