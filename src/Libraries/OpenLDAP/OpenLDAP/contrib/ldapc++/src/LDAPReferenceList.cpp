/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 8, 2025.
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


#include "LDAPReferenceList.h"
#include "LDAPSearchReference.h"

LDAPReferenceList::LDAPReferenceList(){
}

LDAPReferenceList::LDAPReferenceList(const LDAPReferenceList& e){
    m_refs = e.m_refs;
}

LDAPReferenceList::~LDAPReferenceList(){
}

size_t LDAPReferenceList::size() const{
    return m_refs.size();
}

bool LDAPReferenceList::empty() const{
    return m_refs.empty();
}

LDAPReferenceList::const_iterator LDAPReferenceList::begin() const{
    return m_refs.begin();
}

LDAPReferenceList::const_iterator LDAPReferenceList::end() const{
    return m_refs.end();
}

void LDAPReferenceList::addReference(const LDAPSearchReference& e){
    m_refs.push_back(e);
}

