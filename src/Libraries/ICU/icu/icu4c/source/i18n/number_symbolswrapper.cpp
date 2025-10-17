/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 16, 2025.
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

// Â© 2020 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html

#include "unicode/utypes.h"

#if !UCONFIG_NO_FORMATTING

#include "number_microprops.h"
#include "unicode/numberformatter.h"

using namespace icu;
using namespace icu::number;
using namespace icu::number::impl;

SymbolsWrapper::SymbolsWrapper(const SymbolsWrapper &other) {
    doCopyFrom(other);
}

SymbolsWrapper::SymbolsWrapper(SymbolsWrapper &&src) noexcept {
    doMoveFrom(std::move(src));
}

SymbolsWrapper &SymbolsWrapper::operator=(const SymbolsWrapper &other) {
    if (this == &other) {
        return *this;
    }
    doCleanup();
    doCopyFrom(other);
    return *this;
}

SymbolsWrapper &SymbolsWrapper::operator=(SymbolsWrapper &&src) noexcept {
    if (this == &src) {
        return *this;
    }
    doCleanup();
    doMoveFrom(std::move(src));
    return *this;
}

SymbolsWrapper::~SymbolsWrapper() {
    doCleanup();
}

void SymbolsWrapper::setTo(const DecimalFormatSymbols &dfs) {
    doCleanup();
    fType = SYMPTR_DFS;
    fPtr.dfs = new DecimalFormatSymbols(dfs);
}

void SymbolsWrapper::setTo(const NumberingSystem *ns) {
    doCleanup();
    fType = SYMPTR_NS;
    fPtr.ns = ns;
}

void SymbolsWrapper::doCopyFrom(const SymbolsWrapper &other) {
    fType = other.fType;
    switch (fType) {
    case SYMPTR_NONE:
        // No action necessary
        break;
    case SYMPTR_DFS:
        // Memory allocation failures are exposed in copyErrorTo()
        if (other.fPtr.dfs != nullptr) {
            fPtr.dfs = new DecimalFormatSymbols(*other.fPtr.dfs);
        } else {
            fPtr.dfs = nullptr;
        }
        break;
    case SYMPTR_NS:
        // Memory allocation failures are exposed in copyErrorTo()
        if (other.fPtr.ns != nullptr) {
            fPtr.ns = new NumberingSystem(*other.fPtr.ns);
        } else {
            fPtr.ns = nullptr;
        }
        break;
    }
}

void SymbolsWrapper::doMoveFrom(SymbolsWrapper &&src) {
    fType = src.fType;
    switch (fType) {
    case SYMPTR_NONE:
        // No action necessary
        break;
    case SYMPTR_DFS:
        fPtr.dfs = src.fPtr.dfs;
        src.fPtr.dfs = nullptr;
        break;
    case SYMPTR_NS:
        fPtr.ns = src.fPtr.ns;
        src.fPtr.ns = nullptr;
        break;
    }
}

void SymbolsWrapper::doCleanup() {
    switch (fType) {
    case SYMPTR_NONE:
        // No action necessary
        break;
    case SYMPTR_DFS:
        delete fPtr.dfs;
        break;
    case SYMPTR_NS:
        delete fPtr.ns;
        break;
    }
}

bool SymbolsWrapper::isDecimalFormatSymbols() const {
    return fType == SYMPTR_DFS;
}

bool SymbolsWrapper::isNumberingSystem() const {
    return fType == SYMPTR_NS;
}

const DecimalFormatSymbols *SymbolsWrapper::getDecimalFormatSymbols() const {
    U_ASSERT(fType == SYMPTR_DFS);
    return fPtr.dfs;
}

const NumberingSystem *SymbolsWrapper::getNumberingSystem() const {
    U_ASSERT(fType == SYMPTR_NS);
    return fPtr.ns;
}

#endif /* #if !UCONFIG_NO_FORMATTING */
