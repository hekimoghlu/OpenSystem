/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 12, 2021.
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
 * Copyright 2003-2011 The OpenLDAP Foundation, All Rights Reserved.
 * COPYING RESTRICTIONS APPLY, see COPYRIGHT file
 */

#include "debug.h"
#include "LDAPObjClass.h"


LDAPObjClass::LDAPObjClass(){
    DEBUG(LDAP_DEBUG_CONSTRUCT,
            "LDAPObjClass::LDAPObjClass( )" << endl);

    oid = string ();
    desc = string ();
    names = StringList ();
    must = StringList();
    may = StringList();
    sup = StringList();
}

LDAPObjClass::LDAPObjClass (const LDAPObjClass &oc){
    DEBUG(LDAP_DEBUG_CONSTRUCT,
            "LDAPObjClass::LDAPObjClass( )" << endl);

    oid = oc.oid;
    desc = oc.desc;
    names = oc.names;
    must = oc.must;
    may = oc.may;
    kind = oc.kind;
    sup = oc.sup;
}

LDAPObjClass::LDAPObjClass (string oc_item, int flags ) { 

    DEBUG(LDAP_DEBUG_CONSTRUCT,
            "LDAPObjClass::LDAPObjClass( )" << endl);

    LDAPObjectClass *o;
    int ret;
    const char *errp;
    o = ldap_str2objectclass ( oc_item.c_str(), &ret, &errp, flags );

    if (o) {
        this->setNames (o->oc_names);
	this->setDesc (o->oc_desc);
        this->setOid (o->oc_oid);
	this->setKind (o->oc_kind);
        this->setMust (o->oc_at_oids_must);
	this->setMay (o->oc_at_oids_may);
        this->setSup (o->oc_sup_oids);
    }
    // else? -> error
}

LDAPObjClass::~LDAPObjClass() {
    DEBUG(LDAP_DEBUG_DESTROY,"LDAPObjClass::~LDAPObjClass()" << endl);
}

void LDAPObjClass::setKind (int oc_kind) {
    kind = oc_kind;
}
    
void LDAPObjClass::setNames (char **oc_names) {
    names = StringList (oc_names);
}

void LDAPObjClass::setMust (char **oc_must) {
    must = StringList (oc_must);
}

void LDAPObjClass::setMay (char **oc_may) {
    may = StringList (oc_may);
}

void LDAPObjClass::setSup (char **oc_sup) {
    sup = StringList (oc_sup);
}

void LDAPObjClass::setDesc (char *oc_desc) {
    desc = string ();
    if (oc_desc)
	desc = oc_desc;
}

void LDAPObjClass::setOid (char *oc_oid) {
    oid = string ();
    if (oc_oid)
	oid = oc_oid;
}

string LDAPObjClass::getOid() const {
    return oid;
}

string LDAPObjClass::getDesc() const {
    return desc;
}

StringList LDAPObjClass::getNames() const {
    return names;
}

StringList LDAPObjClass::getMust() const {
    return must;
}

StringList LDAPObjClass::getMay() const {
    return may;
}

StringList LDAPObjClass::getSup() const {
    return sup;
}

string LDAPObjClass::getName() const {

    if (names.empty())
	return "";
    else
	return *(names.begin());
}

int LDAPObjClass::getKind() const {
     return kind;
}


