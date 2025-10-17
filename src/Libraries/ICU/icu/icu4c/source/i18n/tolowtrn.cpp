/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 10, 2022.
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
*   Copyright (C) 2001-2007, International Business Machines
*   Corporation and others.  All Rights Reserved.
**********************************************************************
*   Date        Name        Description
*   05/24/01    aliu        Creation.
**********************************************************************
*/

#include "unicode/utypes.h"

#if !UCONFIG_NO_TRANSLITERATION

#include "unicode/uchar.h"
#include "unicode/ustring.h"
#include "tolowtrn.h"
#include "ustr_imp.h"
#include "cpputils.h"

U_NAMESPACE_BEGIN

UOBJECT_DEFINE_RTTI_IMPLEMENTATION(LowercaseTransliterator)

/**
 * Constructs a transliterator.
 */
LowercaseTransliterator::LowercaseTransliterator() : 
    CaseMapTransliterator(UNICODE_STRING("Any-Lower", 9), ucase_toFullLower)
{
}

/**
 * Destructor.
 */
LowercaseTransliterator::~LowercaseTransliterator() {
}

/**
 * Copy constructor.
 */
LowercaseTransliterator::LowercaseTransliterator(const LowercaseTransliterator& o) :
    CaseMapTransliterator(o)
{
}

/**
 * Assignment operator.
 */
/*LowercaseTransliterator& LowercaseTransliterator::operator=(
                             const LowercaseTransliterator& o) {
    CaseMapTransliterator::operator=(o);
    return *this;
}*/

/**
 * Transliterator API.
 */
LowercaseTransliterator* LowercaseTransliterator::clone() const {
    return new LowercaseTransliterator(*this);
}

U_NAMESPACE_END

#endif /* #if !UCONFIG_NO_TRANSLITERATION */
