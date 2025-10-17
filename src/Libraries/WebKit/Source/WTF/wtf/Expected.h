/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 27, 2023.
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
// Implementation of Library Fundamentals v3's std::expected, as described here: http://wg21.link/p0323r4

#pragma once

/*
    expected synopsis

namespace std {
namespace experimental {
inline namespace fundamentals_v3 {
    // ?.?.4, Expected for object types
    template <class T, class E>
        class expected;

    // ?.?.5, Expected specialization for void
    template <class E>
        class expected<void,E>;

    // ?.?.6, unexpect tag
    struct unexpect_t {
       unexpect_t() = default;
    };
    inline constexpr unexpect_t unexpect{};

    // ?.?.7, class bad_expected_access
    template <class E>
       class bad_expected_access;

    // ?.?.8, Specialization for void.
    template <>
       class bad_expected_access<void>;

    // ?.?.9, Expected relational operators
    template <class T, class E>
        constexpr bool operator==(const expected<T, E>&, const expected<T, E>&);
    template <class T, class E>
        constexpr bool operator!=(const expected<T, E>&, const expected<T, E>&);

    // ?.?.10, Comparison with T
    template <class T, class E>
      constexpr bool operator==(const expected<T, E>&, const T&);
    template <class T, class E>
      constexpr bool operator==(const T&, const expected<T, E>&);
    template <class T, class E>
      constexpr bool operator!=(const expected<T, E>&, const T&);
    template <class T, class E>
      constexpr bool operator!=(const T&, const expected<T, E>&);

    // ?.?.10, Comparison with unexpected<E>
    template <class T, class E>
      constexpr bool operator==(const expected<T, E>&, const unexpected<E>&);
    template <class T, class E>
      constexpr bool operator==(const unexpected<E>&, const expected<T, E>&);
    template <class T, class E>
      constexpr bool operator!=(const expected<T, E>&, const unexpected<E>&);
    template <class T, class E>
      constexpr bool operator!=(const unexpected<E>&, const expected<T, E>&);

    // ?.?.11, Specialized algorithms
    void swap(expected<T, E>&, expected<T, E>&) noexcept(see below);

    template <class T, class E>
    class expected
    {
    public:
        typedef T value_type;
        typedef E error_type;
        typedef unexpected<E> unexpected_type;
    
        template <class U>
            struct rebind {
            using type = expected<U, error_type>;
          };
    
        // ?.?.4.1, constructors
        constexpr expected();
        constexpr expected(const expected&);
        constexpr expected(expected&&) noexcept(see below);
        template <class U, class G>
            EXPLICIT constexpr expected(const expected<U, G>&);
        template <class U, class G>
            EXPLICIT constexpr expected(expected<U, G>&&);
    
        template <class U = T>
            EXPLICIT constexpr expected(U&& v);
    
        template <class... Args>
            constexpr explicit expected(in_place_t, Args&&...);
        template <class U, class... Args>
            constexpr explicit expected(in_place_t, initializer_list<U>, Args&&...);
        template <class G = E>
            constexpr expected(unexpected<G> const&);
        template <class G = E>
            constexpr expected(unexpected<G> &&);
        template <class... Args>
            constexpr explicit expected(unexpect_t, Args&&...);
        template <class U, class... Args>
            constexpr explicit expected(unexpect_t, initializer_list<U>, Args&&...);
    
        // ?.?.4.2, destructor
        ~expected();
    
        // ?.?.4.3, assignment
        expected& operator=(const expected&);
        expected& operator=(expected&&) noexcept(see below);
        template <class U = T> expected& operator=(U&&);
        template <class G = E>
            expected& operator=(const unexpected<G>&);
        template <class G = E>
            expected& operator=(unexpected<G>&&) noexcept(see below);
    
        template <class... Args>
            void emplace(Args&&...);
        template <class U, class... Args>
            void emplace(initializer_list<U>, Args&&...);
    
        // ?.?.4.4, swap
        void swap(expected&) noexcept(see below);
    
        // ?.?.4.5, observers
        constexpr const T* operator ->() const;
        constexpr T* operator ->();
        constexpr const T& operator *() const&;
        constexpr T& operator *() &;
        constexpr const T&& operator *() const &&;
        constexpr T&& operator *() &&;
        constexpr explicit operator bool() const noexcept;
        constexpr bool has_value() const noexcept;
        constexpr const T& value() const&;
        constexpr T& value() &;
        constexpr const T&& value() const &&;
        constexpr T&& value() &&;
        constexpr const E& error() const&;
        constexpr E& error() &;
        constexpr const E&& error() const &&;
        constexpr E&& error() &&;
        template <class U>
            constexpr T value_or(U&&) const&;
        template <class U>
            T value_or(U&&) &&;
    
    private:
        bool has_val; // exposition only
        union
        {
            value_type val; // exposition only
            unexpected_type unexpect; // exposition only
        };
    };

}}}

*/

#include <cstdlib>
#include <initializer_list>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>
#include <wtf/Assertions.h>
#include <wtf/Compiler.h>
#include <wtf/StdLibExtras.h>
#include <wtf/Unexpected.h>

namespace std {
namespace experimental {
inline namespace fundamentals_v3 {

struct unexpected_t {
    unexpected_t() = default;
};
#if __cplusplus < 201703L
#define __EXPECTED_INLINE_VARIABLE static const
#else
#define __EXPECTED_INLINE_VARIABLE inline
#endif

__EXPECTED_INLINE_VARIABLE constexpr unexpected_t unexpect { };

template<class E> class bad_expected_access;

template<>
class bad_expected_access<void> : public std::exception {
public:
    explicit bad_expected_access() { }
};

template<class E>
class bad_expected_access : public bad_expected_access<void> {
public:
    explicit bad_expected_access(E val) : val(val) { }
    const char* what() const noexcept override { return std::exception::what(); }
    E& error() & { return val; }
    const E& error() const& { return val; }
    E&& error() && { return WTFMove(val); }
    const E&&  error() const&& { return WTFMove(val); }

private:
    E val;
};

namespace __expected_detail {

#if COMPILER_SUPPORTS(EXCEPTIONS)
#define __EXPECTED_THROW(__exception) (throw __exception)
#else
inline NO_RETURN_DUE_TO_CRASH void __expected_terminate() { RELEASE_ASSERT_NOT_REACHED(); }
#define __EXPECTED_THROW(...) __expected_detail::__expected_terminate()
#endif

__EXPECTED_INLINE_VARIABLE constexpr enum class value_tag_t { } value_tag { };
__EXPECTED_INLINE_VARIABLE constexpr enum class error_tag_t { } error_tag { };

template<class T, class E>
struct base {
    typedef T value_type;
    typedef E error_type;
    typedef unexpected<E> unexpected_type;
    std::variant<value_type, error_type> s;
    constexpr base() { }
    constexpr base(value_tag_t, const value_type& val) : s(std::in_place_index_t<0>(), val) { }
    constexpr base(value_tag_t, value_type&& val) : s(std::in_place_index_t<0>(), std::forward<value_type>(val)) { }
    constexpr base(error_tag_t, const error_type& err) : s(std::in_place_index_t<1>(), err) { }
    constexpr base(error_tag_t, error_type&& err) : s(std::in_place_index_t<1>(), std::forward<error_type>(err)) { }
    constexpr base(const base& o)
        : s(o.s) { }
    constexpr base(base&& o)
        : s(std::forward<std::variant<value_type, error_type>>(o.s)) { }
};

template<class E>
struct voidbase {
    typedef void value_type;
    typedef E error_type;
    typedef unexpected<E> unexpected_type;
    std::optional<error_type> error;
    constexpr voidbase() { }
    constexpr voidbase(const error_type& err) : error(err) { }
    constexpr voidbase(error_type&& err) : error(std::forward<error_type>(err)) { }
    constexpr voidbase(const voidbase& o)
        : error(o.error) { }
    constexpr voidbase(voidbase&& o)
        : error(std::forward<std::optional<error_type>>(o.error)) { }
};

} // namespace __expected_detail

template<class T, class E>
class expected : private __expected_detail::base<T, E> {
    WTF_MAKE_FAST_ALLOCATED;
    typedef __expected_detail::base<T, E> base;

public:
    typedef typename base::value_type value_type;
    typedef typename base::error_type error_type;
    typedef typename base::unexpected_type unexpected_type;

private:
    typedef expected<value_type, error_type> type;

public:
    template<class U> struct rebind {
        using type = expected<U, error_type>;
    };

    constexpr expected() : base() { }
    constexpr expected(const expected&) = default;
    constexpr expected(expected&&) = default;

    constexpr expected(const value_type& e) : base(__expected_detail::value_tag, e) { }
    constexpr expected(value_type&& e) : base(__expected_detail::value_tag, std::forward<value_type>(e)) { }
    template<class... Args> constexpr explicit expected(std::in_place_t, Args&&... args) : base(__expected_detail::value_tag, value_type(std::forward<Args>(args)...)) { }
    // template<class U, class... Args> constexpr explicit expected(in_place_t, std::initializer_list<U>, Args&&...);
    constexpr expected(const unexpected_type& u) : base(__expected_detail::error_tag, u.value()) { }
    constexpr expected(unexpected_type&& u) : base(__expected_detail::error_tag, std::forward<unexpected_type>(u).value()) { }
    template<class Err> constexpr expected(const unexpected<Err>& u) : base(__expected_detail::error_tag, u.value()) { }
    template<class Err> constexpr expected(unexpected<Err>&& u) : base(__expected_detail::error_tag, std::forward<Err>(u.value())) { }
    template<class... Args> constexpr explicit expected(unexpected_t, Args&&... args) : base(__expected_detail::error_tag, error_type(std::forward<Args>(args)...)) { }
    // template<class U, class... Args> constexpr explicit expected(unexpected_t, std::initializer_list<U>, Args&&...);

    ~expected() = default;

    expected& operator=(const expected& e) { type(e).swap(*this); return *this; }
    expected& operator=(expected&& e) { type(WTFMove(e)).swap(*this); return *this; }
    template<class U> expected& operator=(U&& u) { type(std::forward<U>(u)).swap(*this); return *this; }
    expected& operator=(const unexpected_type& u) { type(u).swap(*this); return *this; }
    expected& operator=(unexpected_type&& u) { type(WTFMove(u)).swap(*this); return *this; }
    // template<class... Args> void emplace(Args&&...);
    // template<class U, class... Args> void emplace(std::initializer_list<U>, Args&&...);

    void swap(expected& o)
    {
        std::swap(base::s, o.s);
    }

    constexpr const value_type* operator->() const { return &std::get<0>(base::s); }
    value_type* operator->() { return &std::get<0>(base::s); }
    constexpr const value_type& operator*() const & { return std::get<0>(base::s); }
    value_type& operator*() & { return std::get<0>(base::s); }
    constexpr const value_type&& operator*() const && { return WTFMove(std::get<0>(base::s)); }
    constexpr value_type&& operator*() && { return std::get<0>(base::s); }
    constexpr explicit operator bool() const { return has_value(); }
    constexpr bool has_value() const { return !base::s.index(); }
    constexpr const value_type& value() const & { return std::get<0>(base::s); }
    constexpr value_type& value() & { return std::get<0>(base::s); }
    constexpr const value_type&& value() const && { return WTFMove(std::get<0>(base::s)); }
    constexpr value_type&& value() && { return WTFMove(std::get<0>(base::s)); }
    constexpr const error_type& error() const & { return std::get<1>(base::s); }
    error_type& error() & { return std::get<1>(base::s); }
    constexpr error_type&& error() && { return WTFMove(std::get<1>(base::s)); }
    constexpr const error_type&& error() const && { return WTFMove(std::get<1>(base::s)); }
    template<class U> constexpr value_type value_or(U&& u) const & { return has_value() ? **this : static_cast<value_type>(std::forward<U>(u)); }
    template<class U> value_type value_or(U&& u) && { return has_value() ? WTFMove(**this) : static_cast<value_type>(std::forward<U>(u)); }
};

template<class E>
class expected<void, E> : private __expected_detail::voidbase<E> {
    typedef __expected_detail::voidbase<E> base;

public:
    typedef typename base::value_type value_type;
    typedef typename base::error_type error_type;
    typedef typename base::unexpected_type unexpected_type;

private:
    typedef expected<value_type, error_type> type;

public:
    template<class U> struct rebind {
        using type = expected<U, error_type>;
    };

    constexpr expected() : base() { }
    constexpr expected(const expected&) = default;
    constexpr expected(expected&&) = default;
    // constexpr explicit expected(in_place_t);
    constexpr expected(unexpected_type const& u) : base(u.value()) { }
    constexpr expected(unexpected_type&& u) : base(std::forward<unexpected_type>(u).value()) { }
    template<class Err> constexpr expected(unexpected<Err> const& u) : base(u.value()) { }

    ~expected() = default;

    expected& operator=(const expected& e) { type(e).swap(*this); return *this; }
    expected& operator=(expected&& e) { type(WTFMove(e)).swap(*this); return *this; }
    expected& operator=(const unexpected_type& u) { type(u).swap(*this); return *this; } // Not in the current paper.
    expected& operator=(unexpected_type&& u) { type(WTFMove(u)).swap(*this); return *this; } // Not in the current paper.
    // void emplace();

    void swap(expected& o)
    {
        std::swap(base::error, o.base::error);
    }

    constexpr explicit operator bool() const { return has_value(); }
    constexpr bool has_value() const { return !base::error; }
    void value() const { !has_value() ? __EXPECTED_THROW(bad_expected_access<void>()) : void(); }
    constexpr const E& error() const & { return *base::error; }
    E& error() & { return *base::error; }
    constexpr E&& error() && { return WTFMove(*base::error); }
};

template<class T, class E> constexpr bool operator==(const expected<T, E>& x, const expected<T, E>& y) { return bool(x) == bool(y) && (x ? x.value() == y.value() : x.error() == y.error()); }

template<class E> constexpr bool operator==(const expected<void, E>& x, const expected<void, E>& y) { return bool(x) == bool(y) && (x ? true : x.error() == y.error()); }

template<class T, class E> constexpr bool operator==(const expected<T, E>& x, const T& y) { return x ? *x == y : false; }
template<class T, class E> constexpr bool operator==(const T& x, const expected<T, E>& y) { return y ? x == *y : false; }

template<class T, class E> constexpr bool operator==(const expected<T, E>& x, const unexpected<E>& y) { return x ? false : x.error() == y.value(); }
template<class T, class E> constexpr bool operator==(const unexpected<E>& x, const expected<T, E>& y) { return y ? false : x.value() == y.error(); }

template<typename T, typename E> void swap(expected<T, E>& x, expected<T, E>& y) { x.swap(y); }

}}} // namespace std::experimental::fundamentals_v3

__EXPECTED_INLINE_VARIABLE constexpr auto& unexpect = std::experimental::unexpect;
template<class T, class E> using Expected = std::experimental::expected<T, E>;
