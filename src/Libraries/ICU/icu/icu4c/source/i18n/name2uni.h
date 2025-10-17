/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 23, 2024.
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
*   06/07/01    aliu        Creation.
**********************************************************************
*/
#ifndef NAME2UNI_H
#define NAME2UNI_H

#include "unicode/utypes.h"

#if !UCONFIG_NO_TRANSLITERATION

#include "unicode/translit.h"
#include "unicode/uniset.h"

U_NAMESPACE_BEGIN

/**
 * A transliterator that performs name to character mapping.
 * It recognizes the Perl syntax \N{name}.
 * @author Alan Liu
 */
class NameUnicodeTransliterator : public Transliterator {
public:

    /**
     * Constructs a transliterator.
     * @param adoptedFilter    the filter for this transliterator.
     */
    NameUnicodeTransliterator(UnicodeFilter* adoptedFilter = nullptr);

    /**
     * Destructor.
     */
    virtual ~NameUnicodeTransliterator();

    /**
     * Copy constructor.
     */
    NameUnicodeTransliterator(const NameUnicodeTransliterator&);

    /**
     * Transliterator API.
     * @return    A copy of the object.
     */
    virtual NameUnicodeTransliterator* clone() const override;

    /**
     * ICU "poor man's RTTI", returns a UClassID for the actual class.
     */
    virtual UClassID getDynamicClassID() const override;

    /**
     * ICU "poor man's RTTI", returns a UClassID for this class.
     */
    U_I18N_API static UClassID U_EXPORT2 getStaticClassID();

 protected:

    /**
     * Implements {@link Transliterator#handleTransliterate}.
     * @param text          the buffer holding transliterated and
     *                      untransliterated text
     * @param offset        the start and limit of the text, the position
     *                      of the cursor, and the start and limit of transliteration.
     * @param incremental   if true, assume more text may be coming after
     *                      pos.contextLimit. Otherwise, assume the text is complete.
     */
    virtual void handleTransliterate(Replaceable& text, UTransPosition& offset,
                                     UBool isIncremental) const override;

    /**
     * Set of characters which occur in Unicode character names.
     */
    UnicodeSet legal;
private:
    /**
     * Assignment operator.
     */
    NameUnicodeTransliterator& operator=(const NameUnicodeTransliterator&);
};

U_NAMESPACE_END

#endif /* #if !UCONFIG_NO_TRANSLITERATION */

#endif
