/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 31, 2025.
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

// Â© 2017 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html

// extradata.h
// created: 2017jun04 Markus W. Scherer
// (pulled out of n2builder.cpp)

// Write mappings and compositions in compact form for Normalizer2 "extra data",
// the data that does not fit into the trie itself.

#ifndef __EXTRADATA_H__
#define __EXTRADATA_H__

#include "unicode/utypes.h"

#if !UCONFIG_NO_NORMALIZATION

#include "unicode/errorcode.h"
#include "unicode/unistr.h"
#include "unicode/utf16.h"
#include "hash.h"
#include "norms.h"
#include "toolutil.h"
#include "utrie2.h"
#include "uvectr32.h"

U_NAMESPACE_BEGIN

class ExtraData : public Norms::Enumerator {
public:
    ExtraData(Norms &n, UBool fast);

    void rangeHandler(UChar32 start, UChar32 end, Norm &norm) override;

    UnicodeString maybeNoMappingsOnly;
    UnicodeString maybeNoMappingsAndCompositions;
    UnicodeString maybeYesCompositions;
    UnicodeString yesYesCompositions;
    UnicodeString yesNoMappingsAndCompositions;
    UnicodeString yesNoMappingsOnly;
    UnicodeString noNoMappingsCompYes;
    UnicodeString noNoMappingsCompBoundaryBefore;
    UnicodeString noNoMappingsCompNoMaybeCC;
    UnicodeString noNoMappingsEmpty;

private:
    /**
     * Requires norm.hasMapping().
     * Returns the offset of the "first unit" from the beginning of the extraData for c,
     * not from the beginning of the dataString.
     * That is the same as the length of the optional data
     * for the raw mapping and the ccc/lccc word.
     */
    int32_t writeMapping(UChar32 c, const Norm &norm, UnicodeString &dataString);
    /** Returns the full offset into the dataString of the "first unit" for c. */
    int32_t writeNoNoMapping(UChar32 c, const Norm &norm,
                             UnicodeString &dataString, Hashtable &previousMappings);
    UBool setNoNoDelta(UChar32 c, Norm &norm) const;
    /** Requires norm.combinesFwd(). */
    void writeCompositions(UChar32 c, const Norm &norm, UnicodeString &dataString);
    void writeExtraData(UChar32 c, Norm &norm);

    UBool optimizeFast;
    Hashtable previousNoNoMappingsCompYes;  // If constructed in runtime code, pass in UErrorCode.
    Hashtable previousNoNoMappingsCompBoundaryBefore;
    Hashtable previousNoNoMappingsCompNoMaybeCC;
    Hashtable previousNoNoMappingsEmpty;
};

U_NAMESPACE_END

#endif // #if !UCONFIG_NO_NORMALIZATION

#endif  // __EXTRADATA_H__
