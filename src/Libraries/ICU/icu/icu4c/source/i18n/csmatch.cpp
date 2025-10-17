/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 5, 2025.
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

#include "unicode/utypes.h"

#if !UCONFIG_NO_CONVERSION
#include "unicode/unistr.h"
#include "unicode/ucnv.h"

#include "csmatch.h"

#include "csrecog.h"
#include "inputext.h"

U_NAMESPACE_BEGIN

CharsetMatch::CharsetMatch()
  : textIn(nullptr), confidence(0), fCharsetName(nullptr), fLang(nullptr)
{
    // nothing else to do.
}

void CharsetMatch::set(InputText *input, const CharsetRecognizer *cr, int32_t conf,
                       const char *csName, const char *lang)
{
    textIn = input;
    confidence = conf; 
    fCharsetName = csName;
    fLang = lang;
    if (cr != nullptr) {
        if (fCharsetName == nullptr) {
            fCharsetName = cr->getName();
        }
        if (fLang == nullptr) {
            fLang = cr->getLanguage();
        }
    }
}

const char* CharsetMatch::getName()const
{
    return fCharsetName; 
}

const char* CharsetMatch::getLanguage()const
{
    return fLang; 
}

int32_t CharsetMatch::getConfidence()const
{
    return confidence;
}

int32_t CharsetMatch::getUChars(char16_t *buf, int32_t cap, UErrorCode *status) const
{
    UConverter *conv = ucnv_open(getName(), status);
    int32_t result = ucnv_toUChars(conv, buf, cap, reinterpret_cast<const char*>(textIn->fRawInput), textIn->fRawLength, status);

    ucnv_close(conv);

    return result;
}

U_NAMESPACE_END

#endif
