/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 17, 2022.
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

#ifndef LDAP_SEARCH_RESULTS_H
#define LDAP_SEARCH_RESULTS_H

#include <LDAPEntry.h>
#include <LDAPEntryList.h>
#include <LDAPMessage.h>
#include <LDAPMessageQueue.h>
#include <LDAPReferenceList.h>
#include <LDAPSearchReference.h>

class LDAPResult;

/**
 * The class stores the results of a synchronous SEARCH-Operation 
 */
class LDAPSearchResults{
    public:
        /**
         * Default-Constructor
         */
        LDAPSearchResults();

        /**
         * For internal use only.
         *
         * This method reads Search result entries from a
         * LDAPMessageQueue-object.
         * @param msg The message queue to read
         */
        LDAPResult* readMessageQueue(LDAPMessageQueue* msg);

        /**
         * The method is used by the client-application to read the
         * result entries of the  SEARCH-Operation. Every call of this
         * method returns one entry. If all entries were read it return 0.
         * @throws LDAPReferralException  If a Search Reference was
         *          returned by the server
         * @returns A LDAPEntry-object as a result of a SEARCH-Operation or
         *          0 if no more entries are there to return.
         */
        LDAPEntry* getNext();
    private :
        LDAPEntryList entryList;
        LDAPReferenceList refList;
        LDAPEntryList::const_iterator entryPos;
        LDAPReferenceList::const_iterator refPos;
};
#endif //LDAP_SEARCH_RESULTS_H


