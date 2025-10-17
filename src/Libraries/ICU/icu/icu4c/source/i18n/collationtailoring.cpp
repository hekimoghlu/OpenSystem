/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 25, 2022.
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
* Copyright (C) 2013-2015, International Business Machines
* Corporation and others.  All Rights Reserved.
*******************************************************************************
* collationtailoring.cpp
*
* created on: 2013mar12
* created by: Markus W. Scherer
*/

#include "unicode/utypes.h"

#if !UCONFIG_NO_COLLATION

#include "unicode/udata.h"
#include "unicode/unistr.h"
#include "unicode/ures.h"
#include "unicode/uversion.h"
#include "unicode/uvernum.h"
#include "cmemory.h"
#include "collationdata.h"
#include "collationsettings.h"
#include "collationtailoring.h"
#include "normalizer2impl.h"
#include "uassert.h"
#include "uhash.h"
#include "umutex.h"
#include "utrie2.h"

U_NAMESPACE_BEGIN

CollationTailoring::CollationTailoring(const CollationSettings *baseSettings)
        : data(nullptr), settings(baseSettings),
          actualLocale(""),
          ownedData(nullptr),
          builder(nullptr), memory(nullptr), bundle(nullptr),
          trie(nullptr), unsafeBackwardSet(nullptr),
          maxExpansions(nullptr) {
    if(baseSettings != nullptr) {
        U_ASSERT(baseSettings->reorderCodesLength == 0);
        U_ASSERT(baseSettings->reorderTable == nullptr);
        U_ASSERT(baseSettings->minHighNoReorder == 0);
    } else {
        settings = new CollationSettings();
    }
    if(settings != nullptr) {
        settings->addRef();
    }
    rules.getTerminatedBuffer();  // ensure NUL-termination
    version[0] = version[1] = version[2] = version[3] = 0;
    maxExpansionsInitOnce.reset();
}

CollationTailoring::~CollationTailoring() {
    SharedObject::clearPtr(settings);
    delete ownedData;
    delete builder;
    udata_close(memory);
    ures_close(bundle);
    utrie2_close(trie);
    delete unsafeBackwardSet;
    uhash_close(maxExpansions);
    maxExpansionsInitOnce.reset();
}

UBool
CollationTailoring::ensureOwnedData(UErrorCode &errorCode) {
    if(U_FAILURE(errorCode)) { return false; }
    if(ownedData == nullptr) {
        const Normalizer2Impl *nfcImpl = Normalizer2Factory::getNFCImpl(errorCode);
        if(U_FAILURE(errorCode)) { return false; }
        ownedData = new CollationData(*nfcImpl);
        if(ownedData == nullptr) {
            errorCode = U_MEMORY_ALLOCATION_ERROR;
            return false;
        }
    }
    data = ownedData;
    return true;
}

void
CollationTailoring::makeBaseVersion(const UVersionInfo ucaVersion, UVersionInfo version) {
    version[0] = UCOL_BUILDER_VERSION;
    version[1] = (ucaVersion[0] << 3) + ucaVersion[1];
    version[2] = ucaVersion[2] << 6;
    version[3] = 0;
}

void
CollationTailoring::setVersion(const UVersionInfo baseVersion, const UVersionInfo rulesVersion) {
    version[0] = UCOL_BUILDER_VERSION;
    version[1] = baseVersion[1];
    version[2] = (baseVersion[2] & 0xc0) + ((rulesVersion[0] + (rulesVersion[0] >> 6)) & 0x3f);
    version[3] = (rulesVersion[1] << 3) + (rulesVersion[1] >> 5) + rulesVersion[2] +
            (rulesVersion[3] << 4) + (rulesVersion[3] >> 4);
}

int32_t
CollationTailoring::getUCAVersion() const {
    return (static_cast<int32_t>(version[1]) << 4) | (version[2] >> 6);
}

CollationCacheEntry::~CollationCacheEntry() {
    SharedObject::clearPtr(tailoring);
}

U_NAMESPACE_END

#endif  // !UCONFIG_NO_COLLATION
