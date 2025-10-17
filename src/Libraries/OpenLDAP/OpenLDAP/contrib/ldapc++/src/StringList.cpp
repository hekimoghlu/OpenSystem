/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 22, 2022.
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

#include "StringList.h"
#include "debug.h"

#include <cstdlib>

using namespace std;

StringList::StringList(){
}

StringList::StringList(const StringList& sl){
    m_data= StringList::ListType(sl.m_data);
}

StringList::StringList(char** values){
    if(values == 0){
        m_data=StringList::ListType();
    }else{
        char** i;
        for(i=values; *i != 0; i++){
            m_data.push_back(string(*i));
        }
    }
}

StringList::~StringList(){
    DEBUG(LDAP_DEBUG_TRACE,"StringList::~StringList()" << endl);
}

char** StringList::toCharArray() const{
    if(!empty()){
        char** ret = (char**) malloc(sizeof(char*) * (size()+1));
        StringList::const_iterator i;
        int j=0;
        for(i=begin(); i != end(); i++,j++){
            ret[j]=(char*) malloc(sizeof(char) * (i->size()+1));
            i->copy(ret[j],string::npos);
            ret[j][i->size()]=0;
        }
        ret[size()]=0;
        return ret;
    }else{
        return 0;
    }
}

void StringList::add(const string& value){
    m_data.push_back(value);
}

size_t StringList::size() const{
    return m_data.size();
}

bool StringList::empty() const{
    return m_data.empty();
}

StringList::const_iterator StringList::begin() const{
    return m_data.begin();
}

StringList::const_iterator StringList::end() const{
    return m_data.end();
}

    
void StringList::clear(){
    m_data.clear();
}

