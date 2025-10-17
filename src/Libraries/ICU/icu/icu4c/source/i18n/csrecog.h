/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 9, 2023.
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

#ifndef __CSRECOG_H
#define __CSRECOG_H

#include "unicode/uobject.h"

#if !UCONFIG_NO_CONVERSION

#include "inputext.h"

U_NAMESPACE_BEGIN

class CharsetMatch;

class CharsetRecognizer : public UMemory
{
 public:
    /**
     * Get the IANA name of this charset.
     * Note that some recognizers can recognize more than one charset, but that this API
     * assumes just one name per recognizer.
     * TODO: need to account for multiple names in public API that enumerates over the
     *       known detectable charsets.
     * @return the charset name.
     */
    virtual const char *getName() const = 0;
    
    /**
     * Get the ISO language code for this charset.
     * @return the language code, or <code>null</code> if the language cannot be determined.
     */
    virtual const char *getLanguage() const;
        
    /*
     * Try the given input text against this Charset, and fill in the results object
     * with the quality of the match plus other information related to the match.
     *
     * Return true if the the input bytes are a potential match, and
     * false if the input data is not compatible with, or illegal in this charset.
     */
    virtual UBool match(InputText *textIn, CharsetMatch *results) const = 0;

    virtual ~CharsetRecognizer();
};

U_NAMESPACE_END

#endif
#endif /* __CSRECOG_H */
