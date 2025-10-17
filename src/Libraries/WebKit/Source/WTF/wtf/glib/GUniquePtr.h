/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 29, 2024.
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
#ifndef GUniquePtr_h
#define GUniquePtr_h

#if USE(GLIB)

#include <gio/gio.h>
#include <utility>
#include <wtf/Noncopyable.h>

namespace WTF {

template<typename T>
struct GPtrDeleter {
    void operator()(T* ptr) const { g_free(ptr); }
};

template<typename T>
struct GFreeDeleter {
    void operator()(T* ptr) const { g_free(ptr); }
};

template<typename T, typename U = GPtrDeleter<T>>
using GUniquePtr = std::unique_ptr<T, U>;

#define FOR_EACH_GLIB_DELETER(macro) \
    macro(GError, g_error_free) \
    macro(GList, g_list_free) \
    macro(GSList, g_slist_free) \
    macro(GPatternSpec, g_pattern_spec_free) \
    macro(GDir, g_dir_close) \
    macro(GTimer, g_timer_destroy) \
    macro(GKeyFile, g_key_file_free) \
    macro(char*, g_strfreev) \
    macro(GVariantIter, g_variant_iter_free) \
    macro(GVariantType, g_variant_type_free) \
    macro(GMarkupParseContext, g_markup_parse_context_free)

#define WTF_DEFINE_GPTR_DELETER(typeName, deleterFunc) \
    template<> struct GPtrDeleter<typeName> \
    { \
        void operator() (typeName* ptr) const \
        { \
            deleterFunc(ptr); \
        } \
    };

FOR_EACH_GLIB_DELETER(WTF_DEFINE_GPTR_DELETER)
#undef FOR_EACH_GLIB_DELETER

#define WTF_DEFINE_DEPRECATED_GPTR_DELETER(typeName, deleterFunc) \
    template<> struct GPtrDeleter<typeName> { \
        void operator()(typeName* ptr) const \
        { \
            ALLOW_DEPRECATED_DECLARATIONS_BEGIN; \
            deleterFunc(ptr); \
            ALLOW_DEPRECATED_DECLARATIONS_END; \
        } \
    };

WTF_DEFINE_DEPRECATED_GPTR_DELETER(GValueArray, g_value_array_free)
#undef WTF_DEFINE_DEPRECATED_GPTR_DELETER

template <typename T> class GUniqueOutPtr {
    WTF_MAKE_NONCOPYABLE(GUniqueOutPtr);
public:
    GUniqueOutPtr()
        : m_ptr(nullptr)
    {
    }

    ~GUniqueOutPtr()
    {
        reset();
    }

    T*& outPtr()
    {
        reset();
        return m_ptr;
    }

    T* release()
    {
        return std::exchange(m_ptr, nullptr);
    }

    T& operator*() const
    {
        ASSERT(m_ptr);
        return *m_ptr;
    }

    T* operator->() const
    {
        ASSERT(m_ptr);
        return m_ptr;
    }

    T* get() const { return m_ptr; }

    bool operator!() const { return !m_ptr; }

    // This conversion operator allows implicit conversion to bool but not to other integer types.
    typedef T* GUniqueOutPtr::*UnspecifiedBoolType;
    operator UnspecifiedBoolType() const { return m_ptr ? &GUniqueOutPtr::m_ptr : 0; }

private:
    void reset()
    {
        if (m_ptr) {
            GUniquePtr<T> deletePtr(m_ptr);
            m_ptr = nullptr;
        }
    }

    T* m_ptr;
};

} // namespace WTF

using WTF::GUniquePtr;
using WTF::GUniqueOutPtr;
using WTF::GFreeDeleter;

#endif // USE(GLIB)

#endif // GUniquePtr_h

