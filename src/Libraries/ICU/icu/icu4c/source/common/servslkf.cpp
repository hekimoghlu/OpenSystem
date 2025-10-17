/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 10, 2023.
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

/*
 ******************************************************************
 */

SimpleLocaleKeyFactory::SimpleLocaleKeyFactory(UObject* objToAdopt,
                                               const UnicodeString& locale,
                                               int32_t kind,
                                               int32_t coverage)
  : LocaleKeyFactory(coverage)
  , _obj(objToAdopt)
  , _id(locale)
  , _kind(kind)
{
}

SimpleLocaleKeyFactory::SimpleLocaleKeyFactory(UObject* objToAdopt,
                                               const Locale& locale,
                                               int32_t kind,
                                               int32_t coverage)
  : LocaleKeyFactory(coverage)
  , _obj(objToAdopt)
  , _id()
  , _kind(kind)
{
    LocaleUtility::initNameFromLocale(locale, _id);
}

SimpleLocaleKeyFactory::~SimpleLocaleKeyFactory()
{
  delete _obj;
  _obj = nullptr;
}

UObject*
SimpleLocaleKeyFactory::create(const ICUServiceKey& key, const ICUService* service, UErrorCode& status) const
{
    if (U_SUCCESS(status)) {
        const LocaleKey& lkey = static_cast<const LocaleKey&>(key);
        if (_kind == LocaleKey::KIND_ANY || _kind == lkey.kind()) {
            UnicodeString keyID;
            lkey.currentID(keyID);
            if (_id == keyID) {
                return service->cloneInstance(_obj);
            }
        }
    }
    return nullptr;
}

//UBool
//SimpleLocaleKeyFactory::isSupportedID(const UnicodeString& id, UErrorCode& /* status */) const
//{
//    return id == _id;
//}

void
SimpleLocaleKeyFactory::updateVisibleIDs(Hashtable& result, UErrorCode& status) const
{
    if (U_SUCCESS(status)) {
        if (_coverage & 0x1) {
            result.remove(_id);
        } else {
            result.put(_id, (void*)this, status);
        }
    }
}

#ifdef SERVICE_DEBUG
UnicodeString&
SimpleLocaleKeyFactory::debug(UnicodeString& result) const
{
    LocaleKeyFactory::debug(result);
    result.append((UnicodeString)", id: ");
    result.append(_id);
    result.append((UnicodeString)", kind: ");
    result.append(_kind);
    return result;
}

UnicodeString&
SimpleLocaleKeyFactory::debugClass(UnicodeString& result) const
{
    return result.append((UnicodeString)"SimpleLocaleKeyFactory");
}
#endif

UOBJECT_DEFINE_RTTI_IMPLEMENTATION(SimpleLocaleKeyFactory)

U_NAMESPACE_END

/* !UCONFIG_NO_SERVICE */
#endif


