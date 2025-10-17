/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 30, 2024.
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
/**
 *******************************************************************************
 * Copyright (C) 2001-2014, International Business Machines Corporation and    *
 * others. All Rights Reserved.                                                *
 *******************************************************************************
 *
 *******************************************************************************
 */
#include "unicode/utypes.h"

#if !UCONFIG_NO_SERVICE

#include "unicode/resbund.h"
#include "uresimp.h"
#include "cmemory.h"
#include "servloc.h"
#include "ustrfmt.h"
#include "uhash.h"
#include "charstr.h"
#include "uassert.h"

#define UNDERSCORE_CHAR ((char16_t)0x005f)
#define AT_SIGN_CHAR    ((char16_t)64)
#define PERIOD_CHAR     ((char16_t)46)

U_NAMESPACE_BEGIN

LocaleKey*
LocaleKey::createWithCanonicalFallback(const UnicodeString* primaryID,
                                       const UnicodeString* canonicalFallbackID,
                                       UErrorCode& status)
{
    return LocaleKey::createWithCanonicalFallback(primaryID, canonicalFallbackID, KIND_ANY, status);
}

LocaleKey*
LocaleKey::createWithCanonicalFallback(const UnicodeString* primaryID,
                                       const UnicodeString* canonicalFallbackID,
                                       int32_t kind,
                                       UErrorCode& status)
{
    if (primaryID == nullptr || U_FAILURE(status)) {
        return nullptr;
    }
    UnicodeString canonicalPrimaryID;
    LocaleUtility::canonicalLocaleString(primaryID, canonicalPrimaryID);
    return new LocaleKey(*primaryID, canonicalPrimaryID, canonicalFallbackID, kind);
}

LocaleKey::LocaleKey(const UnicodeString& primaryID,
                     const UnicodeString& canonicalPrimaryID,
                     const UnicodeString* canonicalFallbackID,
                     int32_t kind)
  : ICUServiceKey(primaryID)
  , _kind(kind)
  , _primaryID(canonicalPrimaryID)
  , _fallbackID()
  , _currentID()
{
    _fallbackID.setToBogus();
    if (_primaryID.length() != 0) {
        if (canonicalFallbackID != nullptr && _primaryID != *canonicalFallbackID) {
            _fallbackID = *canonicalFallbackID;
        }
    }

    _currentID = _primaryID;
}

LocaleKey::~LocaleKey() {}

UnicodeString&
LocaleKey::prefix(UnicodeString& result) const {
    if (_kind != KIND_ANY) {
        char16_t buffer[64];
        uprv_itou(buffer, 64, _kind, 10, 0);
        UnicodeString temp(buffer);
        result.append(temp);
    }
    return result;
}

int32_t
LocaleKey::kind() const {
    return _kind;
}

UnicodeString&
LocaleKey::canonicalID(UnicodeString& result) const {
    return result.append(_primaryID);
}

UnicodeString&
LocaleKey::currentID(UnicodeString& result) const {
    if (!_currentID.isBogus()) {
        result.append(_currentID);
    }
    return result;
}

UnicodeString&
LocaleKey::currentDescriptor(UnicodeString& result) const {
    if (!_currentID.isBogus()) {
        prefix(result).append(PREFIX_DELIMITER).append(_currentID);
    } else {
        result.setToBogus();
    }
    return result;
}

Locale&
LocaleKey::canonicalLocale(Locale& result) const {
    return LocaleUtility::initLocaleFromName(_primaryID, result);
}

Locale&
LocaleKey::currentLocale(Locale& result) const {
    return LocaleUtility::initLocaleFromName(_currentID, result);
}

UBool
LocaleKey::fallback() {
    if (!_currentID.isBogus()) {
        int x = _currentID.lastIndexOf(UNDERSCORE_CHAR);
        if (x != -1) {
            _currentID.remove(x); // truncate current or fallback, whichever we're pointing to
            return true;
        }

        if (!_fallbackID.isBogus()) {
            _currentID = _fallbackID;
            _fallbackID.setToBogus();
            return true;
        }

        if (_currentID.length() > 0) {
            _currentID.remove(0); // completely truncate
            return true;
        }

        _currentID.setToBogus();
    }

    return false;
}

UBool
LocaleKey::isFallbackOf(const UnicodeString& id) const {
    UnicodeString temp(id);
    parseSuffix(temp);
    return temp.indexOf(_primaryID) == 0 &&
        (temp.length() == _primaryID.length() ||
        temp.charAt(_primaryID.length()) == UNDERSCORE_CHAR);
}

#ifdef SERVICE_DEBUG
UnicodeString&
LocaleKey::debug(UnicodeString& result) const
{
    ICUServiceKey::debug(result);
    result.append((UnicodeString)" kind: ");
    result.append(_kind);
    result.append((UnicodeString)" primaryID: ");
    result.append(_primaryID);
    result.append((UnicodeString)" fallbackID: ");
    result.append(_fallbackID);
    result.append((UnicodeString)" currentID: ");
    result.append(_currentID);
    return result;
}

UnicodeString&
LocaleKey::debugClass(UnicodeString& result) const
{
    return result.append((UnicodeString)"LocaleKey ");
}
#endif

UOBJECT_DEFINE_RTTI_IMPLEMENTATION(LocaleKey)

U_NAMESPACE_END

/* !UCONFIG_NO_SERVICE */
#endif


