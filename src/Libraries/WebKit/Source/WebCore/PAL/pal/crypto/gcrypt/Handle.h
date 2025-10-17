/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 7, 2024.
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
#pragma once

#include <gcrypt.h>

namespace PAL {
namespace GCrypt {

template<typename T>
struct HandleDeleter {
public:
    void operator()(T handle) = delete;
};

template<typename T>
class Handle {
public:
    Handle() = default;

    explicit Handle(T handle)
        : m_handle(handle)
    { }

    ~Handle()
    {
        clear();
    }

    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;

    Handle(Handle&&) = delete;
    Handle& operator=(Handle&&) = delete;

    void clear()
    {
        if (m_handle)
            HandleDeleter<T>()(m_handle);
        m_handle = nullptr;
    }

    T release()
    {
        T handle = m_handle;
        m_handle = nullptr;
        return handle;
    }

    T* operator&() { return &m_handle; }

    T handle() const { return m_handle; }
    operator T() const { return m_handle; }

    bool operator!() const { return !m_handle; }

private:
    T m_handle { nullptr };
};

template<>
struct HandleDeleter<gcry_cipher_hd_t> {
    void operator()(gcry_cipher_hd_t handle)
    {
        gcry_cipher_close(handle);
    }
};

template<>
struct HandleDeleter<gcry_ctx_t> {
    void operator()(gcry_ctx_t handle)
    {
        gcry_ctx_release(handle);
    }
};

template<>
struct HandleDeleter<gcry_mac_hd_t> {
    void operator()(gcry_mac_hd_t handle)
    {
        gcry_mac_close(handle);
    }
};

template<>
struct HandleDeleter<gcry_mpi_t> {
    void operator()(gcry_mpi_t handle)
    {
        gcry_mpi_release(handle);
    }
};

template<>
struct HandleDeleter<gcry_mpi_point_t> {
    void operator()(gcry_mpi_point_t handle)
    {
        gcry_mpi_point_release(handle);
    }
};

template<>
struct HandleDeleter<gcry_sexp_t> {
    void operator()(gcry_sexp_t handle)
    {
        gcry_sexp_release(handle);
    }
};

} // namespace GCrypt
} // namespace PAL
