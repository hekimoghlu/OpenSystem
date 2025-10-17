/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 25, 2021.
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

#include "LDAPUrlList.h"
#include <assert.h>
#include "debug.h"

using namespace std;

LDAPUrlList::LDAPUrlList(){
    DEBUG(LDAP_DEBUG_CONSTRUCT," LDAPUrlList::LDAPUrlList()" << endl);
    m_urls=LDAPUrlList::ListType();
}

LDAPUrlList::LDAPUrlList(const LDAPUrlList& urls){
    DEBUG(LDAP_DEBUG_CONSTRUCT," LDAPUrlList::LDAPUrlList(&)" << endl);
    m_urls = urls.m_urls;
}


LDAPUrlList::LDAPUrlList(char** url){
    DEBUG(LDAP_DEBUG_CONSTRUCT," LDAPUrlList::LDAPUrlList()" << endl);
    char** i;
    assert(url);
    for(i = url; *i != 0; i++){
        add(LDAPUrl(*i));
    }
}

LDAPUrlList::~LDAPUrlList(){
    DEBUG(LDAP_DEBUG_DESTROY," LDAPUrlList::~LDAPUrlList()" << endl);
    m_urls.clear();
}

size_t LDAPUrlList::size() const{
    return m_urls.size();
}

bool LDAPUrlList::empty() const{
    return m_urls.empty();
}

LDAPUrlList::const_iterator LDAPUrlList::begin() const{
    return m_urls.begin();
}

LDAPUrlList::const_iterator LDAPUrlList::end() const{
    return m_urls.end();
}

void LDAPUrlList::add(const LDAPUrl& url){
    m_urls.push_back(url);
}

