/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 29, 2023.
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

// Â© 2024 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html

#include "unicode/utypes.h"

#ifndef U_HIDE_DEPRECATED_API

#ifndef MESSAGEFORMAT2_UTILS_H
#define MESSAGEFORMAT2_UTILS_H

#if U_SHOW_CPLUSPLUS_API

#if !UCONFIG_NO_FORMATTING

#if !UCONFIG_NO_MF2

#include "unicode/unistr.h"
#include "uvector.h"

U_NAMESPACE_BEGIN

namespace message2 {

    // Helpers

    template<typename T>
    static T* copyArray(const T* source, int32_t len, UErrorCode& status) {
        if (U_FAILURE(status)) {
            return nullptr;
        }
        U_ASSERT(source != nullptr);
        U_ASSERT(len >= 0);
        T* dest = new T[len];
        if (dest == nullptr) {
            status = U_MEMORY_ALLOCATION_ERROR;
        } else {
            for (int32_t i = 0; i < len; i++) {
                dest[i] = source[i];
            }
        }
        return dest;
    }

    template<typename T>
    static T* copyVectorToArray(const UVector& source, UErrorCode& status) {
        if (U_FAILURE(status)) {
            return nullptr;
        }
        int32_t len = source.size();
        T* dest = new T[len];
        if (dest == nullptr) {
            status = U_MEMORY_ALLOCATION_ERROR;
        } else {
            for (int32_t i = 0; i < len; i++) {
                dest[i] = *(static_cast<T*>(source.elementAt(i)));
            }
        }
        return dest;
    }

    template<typename T>
    static T* moveVectorToArray(UVector& source, UErrorCode& status) {
        if (U_FAILURE(status)) {
            return nullptr;
        }

        int32_t len = source.size();
        T* dest = new T[len];
        if (dest == nullptr) {
            status = U_MEMORY_ALLOCATION_ERROR;
        } else {
            for (int32_t i = 0; i < len; i++) {
                dest[i] = std::move(*static_cast<T*>(source.elementAt(i)));
            }
            source.removeAllElements();
        }
        return dest;
    }

    inline UVector* createUVectorNoAdopt(UErrorCode& status) {
        if (U_FAILURE(status)) {
            return nullptr;
        }
        LocalPointer<UVector> result(new UVector(status));
        if (U_FAILURE(status)) {
            return nullptr;
        }
        return result.orphan();
    }

    inline UVector* createUVector(UErrorCode& status) {
        UVector* result = createUVectorNoAdopt(status);
        if (U_FAILURE(status)) {
            return nullptr;
        }
        result->setDeleter(uprv_deleteUObject);
        return result;
    }

    static UBool stringsEqual(const UElement s1, const UElement s2) {
        return (*static_cast<UnicodeString*>(s1.pointer) == *static_cast<UnicodeString*>(s2.pointer));
    }

    inline UVector* createStringUVector(UErrorCode& status) {
        UVector* v = createUVector(status);
        if (U_FAILURE(status)) {
            return nullptr;
        }
        v->setComparer(stringsEqual);
        return v;
    }

    inline UVector* createStringVectorNoAdopt(UErrorCode& status) {
        UVector* v = createUVectorNoAdopt(status);
        if (U_FAILURE(status)) {
            return nullptr;
        }
        v->setComparer(stringsEqual);
        return v;
    }

    template<typename T>
    inline T* create(T&& node, UErrorCode& status) {
        if (U_FAILURE(status)) {
            return nullptr;
        }
        T* result = new T(std::move(node));
        if (result == nullptr) {
            status = U_MEMORY_ALLOCATION_ERROR;
        }
        return result;
    }

} // namespace message2

U_NAMESPACE_END

#endif /* #if !UCONFIG_NO_MF2 */

#endif /* #if !UCONFIG_NO_FORMATTING */

#endif /* U_SHOW_CPLUSPLUS_API */

#endif // MESSAGEFORMAT2_UTILS_H

#endif // U_HIDE_DEPRECATED_API
// eof
