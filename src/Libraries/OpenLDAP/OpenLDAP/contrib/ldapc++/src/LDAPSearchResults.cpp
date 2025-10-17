/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 23, 2025.
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


#include "LDAPException.h"
#include "LDAPSearchResult.h"
#include "LDAPResult.h"

#include "LDAPSearchResults.h"

LDAPSearchResults::LDAPSearchResults(){
    entryPos = entryList.begin();
    refPos = refList.begin();
}

LDAPResult* LDAPSearchResults::readMessageQueue(LDAPMessageQueue* msg){
    if(msg != 0){
        LDAPMsg* res=0;
        for(;;){
            try{
                res = msg->getNext();
            }catch (LDAPException e){
                throw;
            }
            switch(res->getMessageType()){ 
                case LDAPMsg::SEARCH_ENTRY :
                    entryList.addEntry(*((LDAPSearchResult*)res)->getEntry());
                break;
                case LDAPMsg::SEARCH_REFERENCE :
                    refList.addReference(*((LDAPSearchReference*)res));
                break;
                default:
                    entryPos=entryList.begin();
                    refPos=refList.begin();
                    return ((LDAPResult*) res);
            }
            delete res;
            res=0;
        }
    }
    return 0;
}

LDAPEntry* LDAPSearchResults::getNext(){
    if( entryPos != entryList.end() ){
        LDAPEntry* ret= new LDAPEntry(*entryPos);
        entryPos++;
        return ret;
    }
    if( refPos != refList.end() ){
        LDAPUrlList urls= refPos->getUrls();
        refPos++;
        throw(LDAPReferralException(urls));
    }
    return 0;
}

