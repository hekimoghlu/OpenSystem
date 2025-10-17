/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 1, 2025.
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
#include "unicode/translit.h"
#include "unicode/normlzr.h"

using icu::Normalizer;
using icu::Replaceable;
using icu::Transliterator;

class UnaccentTransliterator : public Transliterator {
    
 public:
    
    /**
     * Constructor
     */
    UnaccentTransliterator();

    /**
     * Destructor
     */
    virtual ~UnaccentTransliterator();

    UClassID getDynamicClassID() const override;
    U_I18N_API static UClassID U_EXPORT2 getStaticClassID();

 protected:

    /**
     * Implement Transliterator API
     */
    void handleTransliterate(Replaceable& text,
                             UTransPosition& index,
                             UBool incremental) const override;

 private:

    /**
     * Unaccent a single character using normalizer.
     */
    char16_t unaccent(char16_t c) const;

    Normalizer normalizer;
};
