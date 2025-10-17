/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 26, 2023.
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
#ifndef AuthenticatedValue_h
#define AuthenticatedValue_h

#include <ptrauth.h>

// On arm64e, signs the given pointer with the address of where it is stored.
// Other archs just have a regular pointer
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wptrauth-null-pointers"
template<typename T>
struct AuthenticatedValue
{
};

// Partial specialization for pointer types
template<typename T>
struct AuthenticatedValue<T*>
{
    AuthenticatedValue() {
        this->value = ptrauth_sign_unauthenticated(nullptr, ptrauth_key_process_dependent_data, this);
    }
    ~AuthenticatedValue() = default;
    AuthenticatedValue(const AuthenticatedValue& other) {
        this->value = ptrauth_auth_and_resign(other.value,
                                              ptrauth_key_process_dependent_data, &other,
                                              ptrauth_key_process_dependent_data, this);
    }
    AuthenticatedValue(AuthenticatedValue&& other) {
        this->value = ptrauth_auth_and_resign(other.value,
                                              ptrauth_key_process_dependent_data, &other,
                                              ptrauth_key_process_dependent_data, this);
        other.value = ptrauth_sign_unauthenticated(nullptr, ptrauth_key_process_dependent_data, &other);
    }
    AuthenticatedValue& operator=(const AuthenticatedValue& other) {
        this->value = ptrauth_auth_and_resign(other.value,
                                              ptrauth_key_process_dependent_data, &other,
                                              ptrauth_key_process_dependent_data, this);
        return *this;
    }
    AuthenticatedValue& operator=(AuthenticatedValue&& other) {
        this->value = ptrauth_auth_and_resign(other.value,
                                              ptrauth_key_process_dependent_data, &other,
                                              ptrauth_key_process_dependent_data, this);
        other.value = ptrauth_sign_unauthenticated(nullptr, ptrauth_key_process_dependent_data, &other);
        return *this;
    }

    // Add a few convenience methods for interoperating with values of the given type
    AuthenticatedValue(const T* other) {
        this->value = (void*)ptrauth_sign_unauthenticated(other, ptrauth_key_process_dependent_data, this);
    }
    AuthenticatedValue& operator=(const T* other) {
        this->value = (void*)ptrauth_sign_unauthenticated(other, ptrauth_key_process_dependent_data, this);
        return *this;
    }
    bool operator==(T* other) const {
        return ptrauth_auth_data(this->value, ptrauth_key_process_dependent_data, this) == other;
    }
    bool operator!=(T* other) const {
        return ptrauth_auth_data(this->value, ptrauth_key_process_dependent_data, this) != other;
    }

    bool operator==(T* other) {
        return ptrauth_auth_data(this->value, ptrauth_key_process_dependent_data, this) == other;
    }
    bool operator!=(T* other) {
        return ptrauth_auth_data(this->value, ptrauth_key_process_dependent_data, this) != other;
    }

    T* operator->() {
        return (T*)ptrauth_auth_data(this->value, ptrauth_key_process_dependent_data, this);
    }

    const T* operator->() const {
        return (const T*)ptrauth_auth_data(this->value, ptrauth_key_process_dependent_data, this);
    }

    operator T*() {
        return (T*)ptrauth_auth_data(this->value, ptrauth_key_process_dependent_data, this);
    }

    operator T*() const {
        return (T*)ptrauth_auth_data(this->value, ptrauth_key_process_dependent_data, this);
    }

private:
    void* value;
};
#pragma clang diagnostic pop

#endif /* AuthenticatedValue_h */
