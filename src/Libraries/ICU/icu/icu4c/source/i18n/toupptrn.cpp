/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 11, 2023.
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

#include "unicode/ustring.h"
#include "unicode/uchar.h"
#include "toupptrn.h"
#include "ustr_imp.h"
#include "cpputils.h"

U_NAMESPACE_BEGIN

UOBJECT_DEFINE_RTTI_IMPLEMENTATION(UppercaseTransliterator)

/**
 * Constructs a transliterator.
 */
UppercaseTransliterator::UppercaseTransliterator() :
    CaseMapTransliterator(UNICODE_STRING("Any-Upper", 9), ucase_toFullUpper)
{
}

/**
 * Destructor.
 */
UppercaseTransliterator::~UppercaseTransliterator() {
}

/**
 * Copy constructor.
 */
UppercaseTransliterator::UppercaseTransliterator(const UppercaseTransliterator& o) :
    CaseMapTransliterator(o)
{
}

/**
 * Assignment operator.
 */
/*UppercaseTransliterator& UppercaseTransliterator::operator=(
                             const UppercaseTransliterator& o) {
    CaseMapTransliterator::operator=(o);
    return *this;
}*/

/**
 * Transliterator API.
 */
UppercaseTransliterator* UppercaseTransliterator::clone() const {
    return new UppercaseTransliterator(*this);
}

U_NAMESPACE_END

#endif /* #if !UCONFIG_NO_TRANSLITERATION */
