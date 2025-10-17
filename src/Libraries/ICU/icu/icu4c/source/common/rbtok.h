/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 6, 2024.
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
#ifndef RBTOK_H
#define RBTOK_H

#include "unicode/utypes.h"

/**
 * \file
 * \brief C++ API: Rule Based Tokenizer
 */

#if !UCONFIG_NO_BREAK_ITERATION

#include "unicode/urbtok.h"
#include "unicode/parseerr.h"
#include "rbbidata57.h"
#include "rbbi57.h"


U_NAMESPACE_BEGIN

/** @internal */
struct RBBIDataHeader57;
struct RBBIStateTableRow57;


/**
 *
 * A subclass of RuleBasedBreakIterator57 that adds tokenization functionality.

 * <p>This class is for internal use only by Apple Inc.</p>
 *
 */
class U_COMMON_API RuleBasedTokenizer : public RuleBasedBreakIterator57 {

private:
    /**
     * The row corresponding to the start state
     * @internal
     */
    const RBBIStateTableRow57 *fStartRow;

    /**
     * The merged flag results for accepting states
     * @internal
     */
    int32_t *fStateFlags;

    /**
     * Character categories for the Latin1 subset of Unicode
     * @internal
     */
    int16_t *fLatin1Cat;

public:
    /**
     * Construct a RuleBasedTokenizer from a set of rules supplied as a string.
     * @param rules The break rules to be used.
     * @param parseError  In the event of a syntax error in the rules, provides the location
     *                    within the rules of the problem.
     * @param status Information on any errors encountered.
     * @internal, used by urbtok57.cpp
     */
    RuleBasedTokenizer(const UnicodeString &rules, UParseError &parseErr, UErrorCode &status);

    /**
     * Constructor from a flattened set of RBBI data in uprv_malloc'd memory.
     *             RulesBasedBreakIterators built from a custom set of rules
     *             are created via this constructor; the rules are compiled
     *             into memory, then the break iterator is constructed here.
     *
     *             The break iterator adopts the memory, and will
     *             free it when done.
     * @internal, used by urbtok57.cpp
     */
    RuleBasedTokenizer(uint8_t *data, UErrorCode &status);

    /**
     * Constructor from a flattened set of RBBI data in umemory which need not
     *             be malloced (e.g. it may be a memory-mapped file, etc.).
       *
     *             This version does not adopt the memory, and does not
     *             free it when done.
     * @internal, used by urbtok57.cpp
     */
    enum EDontAdopt {
        kDontAdopt
    };
    RuleBasedTokenizer(const uint8_t *data, enum EDontAdopt dontAdopt, UErrorCode &status);

    /**
     * Destructor
     *  @internal
     */
    virtual ~RuleBasedTokenizer();

    /**
     * Fetch the next set of tokens.
     * @param maxTokens The maximum number of tokens to return.
     * @param outTokenRanges Pointer to output array of token ranges.
     * @param outTokenFlags (optional) pointer to output array of token flags.
     * @internal, used by urbtok57.cpp
     */
    int32_t tokenize(int32_t maxTokens, RuleBasedTokenRange *outTokenRanges, unsigned long *outTokenFlags);

private:
    /**
      * Common initialization function, used by constructors.
      * @internal
      */
    void init();
};

U_NAMESPACE_END

#endif /* #if !UCONFIG_NO_BREAK_ITERATION */

#endif
