/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 15, 2022.
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
#ifndef AAPLBFCT_H
#define AAPLBFCT_H

#include "unicode/utypes.h"
#include "unicode/uobject.h"
#include "unicode/utext.h"
#include "unicode/uscript.h"
#include "brkeng.h"
#include "dictbe.h"

U_NAMESPACE_BEGIN

class AppleLanguageBreakFactory : public ICULanguageBreakFactory {
 public:

  /**
   * <p>Standard constructor.</p>
   *
   */
  AppleLanguageBreakFactory(UErrorCode &status);

  /**
   * <p>Virtual destructor.</p>
   */
  virtual ~AppleLanguageBreakFactory();

 protected:

  /**
   * <p>Create a DictionaryMatcher for the specified script and break type.</p>
   * @param script An ISO 15924 script code that identifies the dictionary to be
   * created.
   * @param breakType The kind of text break for which a dictionary is 
   * sought.
   * @return A DictionaryMatcher with the desired characteristics, or NULL.
   */
  virtual DictionaryMatcher *loadDictionaryMatcherFor(UScriptCode script);

};

U_NAMESPACE_END

    /* AAPLBFCT_H */
#endif
