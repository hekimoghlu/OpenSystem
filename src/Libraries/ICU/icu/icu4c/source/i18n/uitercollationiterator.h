/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 2, 2024.
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
* Copyright (C) 2012-2016, International Business Machines
* Corporation and others.  All Rights Reserved.
*******************************************************************************
* uitercollationiterator.h
*
* created on: 2012sep23 (from utf16collationiterator.h)
* created by: Markus W. Scherer
*/

#ifndef __UITERCOLLATIONITERATOR_H__
#define __UITERCOLLATIONITERATOR_H__

#include "unicode/utypes.h"

#if !UCONFIG_NO_COLLATION

#include "unicode/uiter.h"
#include "cmemory.h"
#include "collation.h"
#include "collationdata.h"
#include "collationiterator.h"
#include "normalizer2impl.h"

U_NAMESPACE_BEGIN

/**
 * UCharIterator-based collation element and character iterator.
 * Handles normalized text inline, with length or NUL-terminated.
 * Unnormalized text is handled by a subclass.
 */
class U_I18N_API UIterCollationIterator : public CollationIterator {
public:
    UIterCollationIterator(const CollationData *d, UBool numeric, UCharIterator &ui)
            : CollationIterator(d, numeric), iter(ui) {}

    virtual ~UIterCollationIterator();

    virtual void resetToOffset(int32_t newOffset) override;

    virtual int32_t getOffset() const override;

    virtual UChar32 nextCodePoint(UErrorCode &errorCode) override;

    virtual UChar32 previousCodePoint(UErrorCode &errorCode) override;

protected:
    virtual uint32_t handleNextCE32(UChar32 &c, UErrorCode &errorCode) override;

    virtual char16_t handleGetTrailSurrogate() override;

    virtual void forwardNumCodePoints(int32_t num, UErrorCode &errorCode) override;

    virtual void backwardNumCodePoints(int32_t num, UErrorCode &errorCode) override;

    UCharIterator &iter;
};

/**
 * Incrementally checks the input text for FCD and normalizes where necessary.
 */
class U_I18N_API FCDUIterCollationIterator : public UIterCollationIterator {
public:
    FCDUIterCollationIterator(const CollationData *data, UBool numeric, UCharIterator &ui, int32_t startIndex)
            : UIterCollationIterator(data, numeric, ui),
              state(ITER_CHECK_FWD), start(startIndex),
              nfcImpl(data->nfcImpl) {}

    virtual ~FCDUIterCollationIterator();

    virtual void resetToOffset(int32_t newOffset) override;

    virtual int32_t getOffset() const override;

    virtual UChar32 nextCodePoint(UErrorCode &errorCode) override;

    virtual UChar32 previousCodePoint(UErrorCode &errorCode) override;

protected:
    virtual uint32_t handleNextCE32(UChar32 &c, UErrorCode &errorCode) override;

    virtual char16_t handleGetTrailSurrogate() override;


    virtual void forwardNumCodePoints(int32_t num, UErrorCode &errorCode) override;

    virtual void backwardNumCodePoints(int32_t num, UErrorCode &errorCode) override;

private:
    /**
     * Switches to forward checking if possible.
     */
    void switchToForward();

    /**
     * Extends the FCD text segment forward or normalizes around pos.
     * @return true if success
     */
    UBool nextSegment(UErrorCode &errorCode);

    /**
     * Switches to backward checking.
     */
    void switchToBackward();

    /**
     * Extends the FCD text segment backward or normalizes around pos.
     * @return true if success
     */
    UBool previousSegment(UErrorCode &errorCode);

    UBool normalize(const UnicodeString &s, UErrorCode &errorCode);

    enum State {
        /**
         * The input text [start..(iter index)[ passes the FCD check.
         * Moving forward checks incrementally.
         * pos & limit are undefined.
         */
        ITER_CHECK_FWD,
        /**
         * The input text [(iter index)..limit[ passes the FCD check.
         * Moving backward checks incrementally.
         * start & pos are undefined.
         */
        ITER_CHECK_BWD,
        /**
         * The input text [start..limit[ passes the FCD check.
         * pos tracks the current text index.
         */
        ITER_IN_FCD_SEGMENT,
        /**
         * The input text [start..limit[ failed the FCD check and was normalized.
         * pos tracks the current index in the normalized string.
         * The text iterator is at the limit index.
         */
        IN_NORM_ITER_AT_LIMIT,
        /**
         * The input text [start..limit[ failed the FCD check and was normalized.
         * pos tracks the current index in the normalized string.
         * The text iterator is at the start index.
         */
        IN_NORM_ITER_AT_START
    };

    State state;

    int32_t start;
    int32_t pos;
    int32_t limit;

    const Normalizer2Impl &nfcImpl;
    UnicodeString normalized;
};

U_NAMESPACE_END

#endif  // !UCONFIG_NO_COLLATION
#endif  // __UITERCOLLATIONITERATOR_H__
