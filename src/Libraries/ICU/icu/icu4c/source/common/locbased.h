/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 4, 2022.
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

// Â© 2016 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html
/*
**********************************************************************
* Copyright (c) 2004-2014, International Business Machines
* Corporation and others.  All Rights Reserved.
**********************************************************************
* Author: Alan Liu
* Created: January 16 2004
* Since: ICU 2.8
**********************************************************************
*/
#ifndef LOCBASED_H
#define LOCBASED_H

#include "unicode/locid.h"
#include "unicode/uobject.h"

/**
 * Macro to declare a locale LocaleBased wrapper object for the given
 * object, which must have two members named `validLocale' and
 * `actualLocale' of size ULOC_FULLNAME_CAPACITY
 */
#define U_LOCALE_BASED(varname, objname) \
  LocaleBased varname((objname).validLocale, (objname).actualLocale)

U_NAMESPACE_BEGIN

/**
 * A utility class that unifies the implementation of getLocale() by
 * various ICU services.  This class is likely to be removed in the
 * ICU 3.0 time frame in favor of an integrated approach with the
 * services framework.
 * @since ICU 2.8
 */
class U_COMMON_API LocaleBased : public UMemory {

 public:

    /**
     * Construct a LocaleBased wrapper around the two pointers.  These
     * will be aliased for the lifetime of this object.
     */
    inline LocaleBased(char* validAlias, char* actualAlias);

    /**
     * Construct a LocaleBased wrapper around the two const pointers.
     * These will be aliased for the lifetime of this object.
     */
    inline LocaleBased(const char* validAlias, const char* actualAlias);

    /**
     * Return locale meta-data for the service object wrapped by this
     * object.  Either the valid or the actual locale may be
     * retrieved.
     * @param type either ULOC_VALID_LOCALE or ULOC_ACTUAL_LOCALE
     * @param status input-output error code
     * @return the indicated locale
     */
    Locale getLocale(ULocDataLocaleType type, UErrorCode& status) const;

    /**
     * Return the locale ID for the service object wrapped by this
     * object.  Either the valid or the actual locale may be
     * retrieved.
     * @param type either ULOC_VALID_LOCALE or ULOC_ACTUAL_LOCALE
     * @param status input-output error code
     * @return the indicated locale ID
     */
    const char* getLocaleID(ULocDataLocaleType type, UErrorCode& status) const;

    /**
     * Set the locale meta-data for the service object wrapped by this
     * object.  If either parameter is zero, it is ignored.
     * @param valid the ID of the valid locale
     * @param actual the ID of the actual locale
     */
    void setLocaleIDs(const char* valid, const char* actual);

    /**
     * Set the locale meta-data for the service object wrapped by this
     * object.
     * @param valid the ID of the valid locale
     * @param actual the ID of the actual locale
     */
    void setLocaleIDs(const Locale& valid, const Locale& actual);

 private:

    char* valid;
    
    char* actual;
};

inline LocaleBased::LocaleBased(char* validAlias, char* actualAlias) :
    valid(validAlias), actual(actualAlias) {
}

inline LocaleBased::LocaleBased(const char* validAlias,
                                const char* actualAlias) :
    // ugh: cast away const
    valid(const_cast<char*>(validAlias)), actual(const_cast<char*>(actualAlias)) {
}

U_NAMESPACE_END

#endif
