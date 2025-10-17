/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 21, 2022.
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
*   Copyright (c) 2002-2011, International Business Machines Corporation
*   and others.  All Rights Reserved.
**********************************************************************
*   Date        Name        Description
*   02/04/2002  aliu        Creation.
**********************************************************************
*/

#ifndef FUNCREPL_H
#define FUNCREPL_H

#include "unicode/utypes.h"

#if !UCONFIG_NO_TRANSLITERATION

#include "unicode/unifunct.h"
#include "unicode/unirepl.h"

U_NAMESPACE_BEGIN

class Transliterator;

/**
 * A replacer that calls a transliterator to generate its output text.
 * The input text to the transliterator is the output of another
 * UnicodeReplacer object.  That is, this replacer wraps another
 * replacer with a transliterator.
 *
 * @author Alan Liu
 */
class FunctionReplacer : public UnicodeFunctor, public UnicodeReplacer {

 private:

    /**
     * The transliterator.  Must not be null.  OWNED.
     */
    Transliterator* translit;

    /**
     * The replacer object.  This generates text that is then
     * processed by 'translit'.  Must not be null.  OWNED.
     */
    UnicodeFunctor* replacer;

 public:

    /**
     * Construct a replacer that takes the output of the given
     * replacer, passes it through the given transliterator, and emits
     * the result as output.
     */
    FunctionReplacer(Transliterator* adoptedTranslit,
                     UnicodeFunctor* adoptedReplacer);

    /**
     * Copy constructor.
     */
    FunctionReplacer(const FunctionReplacer& other);

    /**
     * Destructor
     */
    virtual ~FunctionReplacer();

    /**
     * Implement UnicodeFunctor
     */
    virtual FunctionReplacer* clone() const override;

    /**
     * UnicodeFunctor API.  Cast 'this' to a UnicodeReplacer* pointer
     * and return the pointer.
     */
    virtual UnicodeReplacer* toReplacer() const override;

    /**
     * UnicodeReplacer API
     */
    virtual int32_t replace(Replaceable& text,
                            int32_t start,
                            int32_t limit,
                            int32_t& cursor) override;

    /**
     * UnicodeReplacer API
     */
    virtual UnicodeString& toReplacerPattern(UnicodeString& rule,
                                             UBool escapeUnprintable) const override;

    /**
     * Implement UnicodeReplacer
     */
    virtual void addReplacementSetTo(UnicodeSet& toUnionTo) const override;

    /**
     * UnicodeFunctor API
     */
    virtual void setData(const TransliterationRuleData*) override;

    /**
     * ICU "poor man's RTTI", returns a UClassID for the actual class.
     */
    virtual UClassID getDynamicClassID() const override;

    /**
     * ICU "poor man's RTTI", returns a UClassID for this class.
     */
    static UClassID U_EXPORT2 getStaticClassID();
};

U_NAMESPACE_END

#endif /* #if !UCONFIG_NO_TRANSLITERATION */
#endif

//eof
