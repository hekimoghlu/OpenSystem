/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 31, 2022.
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


#include "LDAPEntryList.h"
#include "LDAPEntry.h"

LDAPEntryList::LDAPEntryList(){
}

LDAPEntryList::LDAPEntryList(const LDAPEntryList& e){
    m_entries = e.m_entries;
}

LDAPEntryList::~LDAPEntryList(){
}

size_t LDAPEntryList::size() const{
    return m_entries.size();
}

bool LDAPEntryList::empty() const{
    return m_entries.empty();
}

LDAPEntryList::const_iterator LDAPEntryList::begin() const{
    return m_entries.begin();
}

LDAPEntryList::const_iterator LDAPEntryList::end() const{
    return m_entries.end();
}

void LDAPEntryList::addEntry(const LDAPEntry& e){
    m_entries.push_back(e);
}

