/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 20, 2025.
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
#include "LDAPControlSet.h"

using namespace std;

LDAPControlSet::LDAPControlSet(){
}

LDAPControlSet::LDAPControlSet(const LDAPControlSet& cs){
    DEBUG(LDAP_DEBUG_CONSTRUCT,"LDAPControlSet::LDAPControlSet(&)" << endl);
    data=cs.data;
}

LDAPControlSet::LDAPControlSet(LDAPControl** controls){
    DEBUG(LDAP_DEBUG_CONSTRUCT,"LDAPControlSet::LDAPControlSet()" << endl);
    if(controls != 0){
        LDAPControl** i;
        for( i=controls; *i!=0;i++) {
            add(LDAPCtrl(*i));
        }
    }
}

LDAPControlSet::~LDAPControlSet(){
    DEBUG(LDAP_DEBUG_DESTROY,"LDAPControlSet::~LDAPControlSet()" << endl);
}

size_t LDAPControlSet::size() const {
    DEBUG(LDAP_DEBUG_TRACE,"LDAPControlSet::size()" << endl);
    return data.size();
}

bool LDAPControlSet::empty() const {
    DEBUG(LDAP_DEBUG_TRACE,"LDAPControlSet::empty()" << endl);
    return data.empty();
}

LDAPControlSet::const_iterator LDAPControlSet::begin() const{
    DEBUG(LDAP_DEBUG_TRACE,"LDAPControlSet::begin()" << endl);
    return data.begin();
}


LDAPControlSet::const_iterator LDAPControlSet::end() const{
    DEBUG(LDAP_DEBUG_TRACE,"LDAPControlSet::end()" << endl);
    return data.end ();
}

void LDAPControlSet::add(const LDAPCtrl& ctrl){
    DEBUG(LDAP_DEBUG_TRACE,"LDAPControlSet::add()" << endl);
    data.push_back(ctrl);
}

LDAPControl** LDAPControlSet::toLDAPControlArray() const{
    DEBUG(LDAP_DEBUG_TRACE, "LDAPControlSet::toLDAPControlArray()" << endl);
    if(data.empty()){
        return 0;
    }else{
        LDAPControl** ret= new LDAPControl*[data.size()+1];
        CtrlList::const_iterator i;
        int j=0;
        for(i=data.begin(); i!=data.end(); i++,j++){
            ret[j] = i->getControlStruct();
        }
        ret[data.size()]=0;
        return ret;
    }
}

void LDAPControlSet::freeLDAPControlArray(LDAPControl **ctrl){
    DEBUG(LDAP_DEBUG_TRACE, "LDAPControlSet::freeLDAPControlArray()" << endl);
    if( ctrl ){
        for( LDAPControl **i = ctrl; *i != 0; ++i ){
	    LDAPCtrl::freeLDAPControlStruct(*i);
	}
    }
    delete[] ctrl;
}
