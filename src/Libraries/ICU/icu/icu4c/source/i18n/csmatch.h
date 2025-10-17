/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 21, 2025.
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
 *   Copyright (C) 2005-2012, International Business Machines
 *   Corporation and others.  All Rights Reserved.
 **********************************************************************
 */

#ifndef __CSMATCH_H
#define __CSMATCH_H

#include "unicode/uobject.h"

#if !UCONFIG_NO_CONVERSION

U_NAMESPACE_BEGIN

class InputText;
class CharsetRecognizer;

/*
 * CharsetMatch represents the results produced by one Charset Recognizer for one input text
 *              Any confidence > 0 indicates a possible match, meaning that the input bytes
 *              are at least legal.
 *
 *              The full results of a detect are represented by an array of these
 *              CharsetMatch objects, each representing a possible matching charset.
 *
 *              Note that a single charset recognizer may detect multiple closely related
 *              charsets, and set different names depending on the exact input bytes seen.
 */
class CharsetMatch : public UMemory
{
 private:
    InputText               *textIn;
    int32_t                  confidence;
    const char              *fCharsetName;
    const char              *fLang;

 public:
    CharsetMatch();

    /**
      * fully set the state of this CharsetMatch.
      * Called by the CharsetRecognizers to record match results.
      * Default (nullptr) parameters for names will be filled by calling the
      *   corresponding getters on the recognizer.
      */
    void set(InputText               *input, 
             const CharsetRecognizer *cr, 
             int32_t                  conf, 
             const char              *csName=nullptr,
             const char              *lang=nullptr);

    /**
      * Return the name of the charset for this Match
      */
    const char *getName() const;

    const char *getLanguage()const;

    int32_t getConfidence()const;

    int32_t getUChars(char16_t *buf, int32_t cap, UErrorCode *status) const;
};

U_NAMESPACE_END

#endif
#endif /* __CSMATCH_H */
