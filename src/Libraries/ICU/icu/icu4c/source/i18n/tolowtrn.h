/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 20, 2024.
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
#ifndef TOLOWTRN_H
#define TOLOWTRN_H

#include "unicode/utypes.h"

#if !UCONFIG_NO_TRANSLITERATION

#include "unicode/translit.h"
#include "casetrn.h"

U_NAMESPACE_BEGIN

/**
 * A transliterator that performs locale-sensitive toLower()
 * case mapping.
 * @author Alan Liu
 */
class LowercaseTransliterator : public CaseMapTransliterator {

 public:

    /**
     * Constructs a transliterator.
     * @param loc the given locale.
     */
    LowercaseTransliterator();

    /**
     * Destructor.
     */
    virtual ~LowercaseTransliterator();

    /**
     * Copy constructor.
     */
    LowercaseTransliterator(const LowercaseTransliterator&);

    /**
     * Transliterator API.
     * @return a copy of the object.
     */
    virtual LowercaseTransliterator* clone() const override;

    /**
     * ICU "poor man's RTTI", returns a UClassID for the actual class.
     */
    virtual UClassID getDynamicClassID() const override;

    /**
     * ICU "poor man's RTTI", returns a UClassID for this class.
     */
    U_I18N_API static UClassID U_EXPORT2 getStaticClassID();
private:

    /**
     * Assignment operator.
     */
    LowercaseTransliterator& operator=(const LowercaseTransliterator&);
};

U_NAMESPACE_END

#endif /* #if !UCONFIG_NO_TRANSLITERATION */

#endif
