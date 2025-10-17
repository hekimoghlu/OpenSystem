/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 26, 2024.
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


#ifndef LDAP_EXCEPTION_H
#define LDAP_EXCEPTION_H

#include <iostream>
#include <string>
#include <stdexcept>

#include <LDAPUrlList.h>

class LDAPAsynConnection;

/**
 * This class is only thrown as an Exception and used to signalize error
 * conditions during LDAP-operations
 */
class LDAPException : public std::runtime_error
{
		
    public :
        /**
         * Constructs a LDAPException-object from the parameters
         * @param res_code A valid LDAP result code.
         * @param err_string    An addional error message for the error
         *                      that happend (optional)
         */
        LDAPException(int res_code, 
                const std::string& err_string=std::string()) throw();
		
        /**
         * Constructs a LDAPException-object from the error state of a
         * LDAPAsynConnection-object
         * @param lc A LDAP-Connection for that an error has happend. The
         *          Constructor tries to read its error state.
         */
        LDAPException(const LDAPAsynConnection *lc) throw();

        /**
         * Destructor
         */
        virtual ~LDAPException() throw();

        /**
         * @return The Result code of the object
         */
        int getResultCode() const throw();

        /**
         * @return The error message that is corresponding to the result
         *          code .
         */
        const std::string& getResultMsg() const throw();
        
        /**
         * @return The addional error message of the error (if it was set)
         */
        const std::string& getServerMsg() const throw();

        
        virtual const char* what() const throw();

        /**
         * This method can be used to dump the data of a LDAPResult-Object.
         * It is only useful for debugging purposes at the moment
         */
        friend std::ostream& operator << (std::ostream &s, LDAPException e) throw();

    private :
        int m_res_code;
        std::string m_res_string;
        std::string m_err_string;
};

/**
 * This class extends LDAPException and is used to signalize Referrals
 * there were received during synchronous LDAP-operations
 */
class LDAPReferralException : public LDAPException
{

    public :
        /**
         * Creates an object that is initialized with a list of URLs
         */
        LDAPReferralException(const LDAPUrlList& urls) throw();

        /**
         * Destructor
         */
        ~LDAPReferralException() throw();

        /**
         * @return The List of URLs of the Referral/Search Reference
         */
        const LDAPUrlList& getUrls() throw();

    private :
        LDAPUrlList m_urlList;
};

#endif //LDAP_EXCEPTION_H
