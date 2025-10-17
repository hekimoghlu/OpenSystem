/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 26, 2024.
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

// Â© 2021 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html

#ifndef LSTMBE_H
#define LSTMBE_H

#include "unicode/utypes.h"

#if !UCONFIG_NO_BREAK_ITERATION

#include "unicode/uniset.h"
#include "unicode/ures.h"
#include "unicode/utext.h"
#include "unicode/utypes.h"

#include "brkeng.h"
#include "dictbe.h"
#include "uvectr32.h"

U_NAMESPACE_BEGIN

class Vectorizer;
struct LSTMData;

/*******************************************************************
 * LSTMBreakEngine
 */

/**
 * <p>LSTMBreakEngine is a kind of DictionaryBreakEngine that uses a
 * LSTM to determine language-specific breaks.</p>
 *
 * <p>After it is constructed a LSTMBreakEngine may be shared between
 * threads without synchronization.</p>
 */
class LSTMBreakEngine : public DictionaryBreakEngine {
public:
    /**
     * <p>Constructor.</p>
     */
    LSTMBreakEngine(const LSTMData* data, const UnicodeSet& set, UErrorCode &status);

    /**
     * <p>Virtual destructor.</p>
     */
    virtual ~LSTMBreakEngine();

    virtual const char16_t* name() const;

protected:
    /**
     * <p>Divide up a range of known dictionary characters handled by this break engine.</p>
     *
     * @param text A UText representing the text
     * @param rangeStart The start of the range of dictionary characters
     * @param rangeEnd The end of the range of dictionary characters
     * @param foundBreaks Output of C array of int32_t break positions, or 0
     * @param status Information on any errors encountered.
     * @return The number of breaks found
     */
     virtual int32_t divideUpDictionaryRange(UText *text,
                                             int32_t rangeStart,
                                             int32_t rangeEnd,
                                             UVector32 &foundBreaks,
                                             UBool isPhraseBreaking,
                                             UErrorCode& status) const override;
private:
    const LSTMData* fData;
    const Vectorizer* fVectorizer;
};

U_CAPI const LanguageBreakEngine* U_EXPORT2 CreateLSTMBreakEngine(
    UScriptCode script, const LSTMData* data, UErrorCode& status);

U_CAPI const LSTMData* U_EXPORT2 CreateLSTMData(
    UResourceBundle* rb, UErrorCode& status);

U_CAPI const LSTMData* U_EXPORT2 CreateLSTMDataForScript(
    UScriptCode script, UErrorCode& status);

U_CAPI void U_EXPORT2 DeleteLSTMData(const LSTMData* data);
U_CAPI const char16_t* U_EXPORT2 LSTMDataName(const LSTMData* data);

U_NAMESPACE_END

#endif /* #if !UCONFIG_NO_BREAK_ITERATION */

#endif  /* LSTMBE_H */
