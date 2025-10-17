/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 3, 2022.
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

// Â© 2018 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html

#ifndef __PLURALRANGES_H__
#define __PLURALRANGES_H__

#include "unicode/utypes.h"

#if !UCONFIG_NO_FORMATTING

#include "unicode/uobject.h"
#include "unicode/locid.h"
#include "unicode/plurrule.h"
#include "standardplural.h"
#include "cmemory.h"

U_NAMESPACE_BEGIN

// Forward declarations
namespace number::impl {
class UFormattedNumberRangeData;
}

class StandardPluralRanges : public UMemory {
  public:
    /** Create a new StandardPluralRanges for the given locale */
    static StandardPluralRanges forLocale(const Locale& locale, UErrorCode& status);

    /** Explicit copy constructor */
    StandardPluralRanges copy(UErrorCode& status) const;

    /** Create an object (called on an rvalue) */
    LocalPointer<StandardPluralRanges> toPointer(UErrorCode& status) && noexcept;

    /** Select rule based on the first and second forms */
    StandardPlural::Form resolve(StandardPlural::Form first, StandardPlural::Form second) const;

    /** Used for data loading. */
    void addPluralRange(
        StandardPlural::Form first,
        StandardPlural::Form second,
        StandardPlural::Form result);

    /** Used for data loading. */
    void setCapacity(int32_t length, UErrorCode& status);

  private:
    struct StandardPluralRangeTriple {
        StandardPlural::Form first;
        StandardPlural::Form second;
        StandardPlural::Form result;
    };

    // TODO: An array is simple here, but it results in linear lookup time.
    // Certain locales have 20-30 entries in this list.
    // Consider changing to a smarter data structure.
    typedef MaybeStackArray<StandardPluralRangeTriple, 3> PluralRangeTriples;
    PluralRangeTriples fTriples;
    int32_t fTriplesLen = 0;
};

U_NAMESPACE_END

#endif /* #if !UCONFIG_NO_FORMATTING */
#endif //__PLURALRANGES_H__
