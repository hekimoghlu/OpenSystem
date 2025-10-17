/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 12, 2023.
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

#ifndef LDAP_REBIND_H
#define LDAP_REBIND_H

#include <string>
#include <LDAPRebindAuth.h>

/**
 * Just an abstract class to provide a mechnism for rebind to another
 * server when chasing referrals. Clients have to implement a class
 * derived from this. To use authentication other than anonymous for
 * referral chasing
 */

class LDAPRebind{
    public:
        virtual ~LDAPRebind() {}
        virtual LDAPRebindAuth* getRebindAuth(const std::string& hostname, 
                int port) const = 0;
};
#endif //LDAP_REBIND_H

