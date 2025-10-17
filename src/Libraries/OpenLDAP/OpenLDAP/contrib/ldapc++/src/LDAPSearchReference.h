/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 27, 2023.
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


#ifndef LDAP_SEARCH_REFERENCE_H
#define LDAP_SEARCH_REFERENCE_H 

#include <LDAPMessage.h>
#include <LDAPUrlList.h>

class LDAPRequest;
class LDAPUrl;

/**
 * This class is used to represent Continuation References that were
 * returned during a SEARCH-Operation.
 */
class LDAPSearchReference : public LDAPMsg{

    public :
        /**
         * Constructor that create an object from the C-API structures
         */
        LDAPSearchReference(const LDAPRequest* req, LDAPMessage* msg);

        /**
         * The Destructor
         */
        ~LDAPSearchReference();

        /**
         * @returns The destination URLs that were send with this message
         */
        const LDAPUrlList& getUrls() const;

    private :
        LDAPUrlList m_urlList;
        LDAPSearchReference();
};



#endif //LDAP_SEARCH_REFERENCE_H
