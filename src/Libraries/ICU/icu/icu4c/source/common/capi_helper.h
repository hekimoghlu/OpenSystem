/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 16, 2023.
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

// Â© 2018 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html

#ifndef __CAPI_HELPER_H__
#define __CAPI_HELPER_H__

#include "unicode/utypes.h"

U_NAMESPACE_BEGIN

/**
 * An internal helper class to help convert between C and C++ APIs.
 */
template<typename CType, typename CPPType, int32_t kMagic>
class IcuCApiHelper {
  public:
    /**
     * Convert from the C type to the C++ type (const version).
     */
    static const CPPType* validate(const CType* input, UErrorCode& status);

    /**
     * Convert from the C type to the C++ type (non-const version).
     */
    static CPPType* validate(CType* input, UErrorCode& status);

    /**
     * Convert from the C++ type to the C type (const version).
     */
    const CType* exportConstForC() const;

    /**
     * Convert from the C++ type to the C type (non-const version).
     */
    CType* exportForC();

    /**
     * Invalidates the object.
     */
    ~IcuCApiHelper();

  private:
    /**
     * While the object is valid, fMagic equals kMagic.
     */
    int32_t fMagic = kMagic;
};


template<typename CType, typename CPPType, int32_t kMagic>
const CPPType*
IcuCApiHelper<CType, CPPType, kMagic>::validate(const CType* input, UErrorCode& status) {
    if (U_FAILURE(status)) {
        return nullptr;
    }
    if (input == nullptr) {
        status = U_ILLEGAL_ARGUMENT_ERROR;
        return nullptr;
    }
    auto* impl = reinterpret_cast<const CPPType*>(input);
    if (static_cast<const IcuCApiHelper<CType, CPPType, kMagic>*>(impl)->fMagic != kMagic) {
        status = U_INVALID_FORMAT_ERROR;
        return nullptr;
    }
    return impl;
}

template<typename CType, typename CPPType, int32_t kMagic>
CPPType*
IcuCApiHelper<CType, CPPType, kMagic>::validate(CType* input, UErrorCode& status) {
    auto* constInput = static_cast<const CType*>(input);
    auto* validated = validate(constInput, status);
    return const_cast<CPPType*>(validated);
}

template<typename CType, typename CPPType, int32_t kMagic>
const CType*
IcuCApiHelper<CType, CPPType, kMagic>::exportConstForC() const {
    return reinterpret_cast<const CType*>(static_cast<const CPPType*>(this));
}

template<typename CType, typename CPPType, int32_t kMagic>
CType*
IcuCApiHelper<CType, CPPType, kMagic>::exportForC() {
    return reinterpret_cast<CType*>(static_cast<CPPType*>(this));
}

template<typename CType, typename CPPType, int32_t kMagic>
IcuCApiHelper<CType, CPPType, kMagic>::~IcuCApiHelper() {
    // head off application errors by preventing use of of deleted objects.
    fMagic = 0;
}


U_NAMESPACE_END

#endif // __CAPI_HELPER_H__
