/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 20, 2024.
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
*******************************************************************************
*
*   Copyright (C) 2001-2008, International Business Machines
*   Corporation and others.  All Rights Reserved.
*
*******************************************************************************
*   file name:  casetrn.h
*   encoding:   UTF-8
*   tab size:   8 (not used)
*   indentation:4
*
*   created on: 2004sep03
*   created by: Markus W. Scherer
*
*   Implementation class for lower-/upper-/title-casing transliterators.
*/

#ifndef __CASETRN_H__
#define __CASETRN_H__

#include "unicode/utypes.h"

#if !UCONFIG_NO_TRANSLITERATION

#include "unicode/translit.h"
#include "ucase.h"

U_NAMESPACE_BEGIN

/**
 * A transliterator that performs locale-sensitive
 * case mapping.
 */
class CaseMapTransliterator : public Transliterator {
public:
    /**
     * Constructs a transliterator.
     * @param loc the given locale.
     * @param id  the transliterator ID.
     * @param map the full case mapping function (see ucase.h)
     */
    CaseMapTransliterator(const UnicodeString &id, UCaseMapFull *map);

    /**
     * Destructor.
     */
    virtual ~CaseMapTransliterator();

    /**
     * Copy constructor.
     */
    CaseMapTransliterator(const CaseMapTransliterator&);

    /**
     * Transliterator API.
     * @return a copy of the object.
     */
    virtual CaseMapTransliterator* clone() const override = 0;

    /**
     * ICU "poor man's RTTI", returns a UClassID for the actual class.
     */
    //virtual UClassID getDynamicClassID() const;

    /**
     * ICU "poor man's RTTI", returns a UClassID for this class.
     */
    U_I18N_API static UClassID U_EXPORT2 getStaticClassID();

protected:
    /**
     * Implements {@link Transliterator#handleTransliterate}.
     * @param text        the buffer holding transliterated and
     *                    untransliterated text
     * @param offset      the start and limit of the text, the position
     *                    of the cursor, and the start and limit of transliteration.
     * @param incremental if true, assume more text may be coming after
     *                    pos.contextLimit.  Otherwise, assume the text is complete.
     */
    virtual void handleTransliterate(Replaceable& text,
                                     UTransPosition& offsets, 
                                     UBool isIncremental) const override;

    UCaseMapFull *fMap;

private:
    /**
     * Assignment operator.
     */
    CaseMapTransliterator& operator=(const CaseMapTransliterator&);

};

U_NAMESPACE_END

/** case context iterator using a Replaceable. This must be a C function because it is a callback. */
U_CFUNC UChar32 U_CALLCONV
utrans_rep_caseContextIterator(void *context, int8_t dir);

#endif /* #if !UCONFIG_NO_TRANSLITERATION */

#endif
