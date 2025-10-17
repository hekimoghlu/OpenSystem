/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 25, 2022.
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
#include "locbased.h"
#include "cstring.h"

U_NAMESPACE_BEGIN

Locale LocaleBased::getLocale(ULocDataLocaleType type, UErrorCode& status) const {
    const char* id = getLocaleID(type, status);
    return Locale(id != nullptr ? id : "");
}

const char* LocaleBased::getLocaleID(ULocDataLocaleType type, UErrorCode& status) const {
    if (U_FAILURE(status)) {
        return nullptr;
    }

    switch(type) {
    case ULOC_VALID_LOCALE:
        return valid;
    case ULOC_ACTUAL_LOCALE:
        return actual;
    default:
        status = U_ILLEGAL_ARGUMENT_ERROR;
        return nullptr;
    }
}

void LocaleBased::setLocaleIDs(const char* validID, const char* actualID) {
    if (validID != nullptr) {
      uprv_strncpy(valid, validID, ULOC_FULLNAME_CAPACITY);
      valid[ULOC_FULLNAME_CAPACITY-1] = 0; // always terminate
    }
    if (actualID != nullptr) {
      uprv_strncpy(actual, actualID, ULOC_FULLNAME_CAPACITY);
      actual[ULOC_FULLNAME_CAPACITY-1] = 0; // always terminate
    }
}

void LocaleBased::setLocaleIDs(const Locale& validID, const Locale& actualID) {
  uprv_strcpy(valid, validID.getName());
  uprv_strcpy(actual, actualID.getName());
}

U_NAMESPACE_END
