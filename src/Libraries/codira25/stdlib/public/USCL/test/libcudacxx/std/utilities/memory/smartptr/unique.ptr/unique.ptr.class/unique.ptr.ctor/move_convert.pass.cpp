/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 7, 2024.
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
// UNSUPPORTED: nvrtc

// <memory>

// unique_ptr

// Test unique_ptr converting move ctor

#include <uscl/std/__memory_>
#include <uscl/std/cassert>

#include "test_macros.h"
#include "type_id.h"
#include "unique_ptr_test_helper.h"

template <int ID = 0>
struct GenericDeleter
{
  __host__ __device__ TEST_CONSTEXPR_CXX23 void operator()(void*) const {}
};

template <int ID = 0>
struct GenericConvertingDeleter
{
  template <int OID>
  __host__ __device__ TEST_CONSTEXPR_CXX23 GenericConvertingDeleter(GenericConvertingDeleter<OID>)
  {}
  __host__ __device__ TEST_CONSTEXPR_CXX23 void operator()(void*) const {}
};

template <class Templ, class Other>
struct is_specialization;

template <template <int> class Templ, int ID1, class Other>
struct is_specialization<Templ<ID1>, Other> : cuda::std::false_type
{};

template <template <int> class Templ, int ID1, int ID2>
struct is_specialization<Templ<ID1>, Templ<ID2>> : cuda::std::true_type
{};

template <class Templ, class Other>
using EnableIfSpecialization =
  typename cuda::std::enable_if<is_specialization<Templ, typename cuda::std::decay<Other>::type>::value>::type;

template <int ID>
struct TrackingDeleter
{
  __host__ __device__ TEST_CONSTEXPR_CXX23 TrackingDeleter()
      : arg_type(&makeArgumentID<>())
  {}

  __host__ __device__ TEST_CONSTEXPR_CXX23 TrackingDeleter(TrackingDeleter const&)
      : arg_type(&makeArgumentID<TrackingDeleter const&>())
  {}

  __host__ __device__ TEST_CONSTEXPR_CXX23 TrackingDeleter(TrackingDeleter&&)
      : arg_type(&makeArgumentID<TrackingDeleter&&>())
  {}

  template <class T, class = EnableIfSpecialization<TrackingDeleter, T>>
  __host__ __device__ TEST_CONSTEXPR_CXX23 TrackingDeleter(T&&)
      : arg_type(&makeArgumentID<T&&>())
  {}

  __host__ __device__ TEST_CONSTEXPR_CXX23 TrackingDeleter& operator=(TrackingDeleter const&)
  {
    arg_type = &makeArgumentID<TrackingDeleter const&>();
    return *this;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX23 TrackingDeleter& operator=(TrackingDeleter&&)
  {
    arg_type = &makeArgumentID<TrackingDeleter&&>();
    return *this;
  }

  template <class T, class = EnableIfSpecialization<TrackingDeleter, T>>
  __host__ __device__ TEST_CONSTEXPR_CXX23 TrackingDeleter& operator=(T&&)
  {
    arg_type = &makeArgumentID<T&&>();
    return *this;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX23 void operator()(void*) const {}

public:
  __host__ __device__ TEST_CONSTEXPR_CXX23 TypeID const* reset() const
  {
    TypeID const* tmp = arg_type;
    arg_type          = nullptr;
    return tmp;
  }

  mutable TypeID const* arg_type;
};

template <class ExpectT, int ID>
__host__ __device__ bool checkArg(TrackingDeleter<ID> const& d)
{
  return d.arg_type && *d.arg_type == makeArgumentID<ExpectT>();
}

template <bool IsArray>
__host__ __device__ TEST_CONSTEXPR_CXX23 void test_sfinae()
{
  typedef typename cuda::std::conditional<IsArray, A[], A>::type VT;

  { // Test that different non-reference deleter types are allowed so long
    // as they convert to each other.
    using U1 = cuda::std::unique_ptr<VT, GenericConvertingDeleter<0>>;
    using U2 = cuda::std::unique_ptr<VT, GenericConvertingDeleter<1>>;
    static_assert(cuda::std::is_constructible<U1, U2&&>::value, "");
  }
  { // Test that different non-reference deleter types are disallowed when
    // they cannot convert.
    using U1 = cuda::std::unique_ptr<VT, GenericDeleter<0>>;
    using U2 = cuda::std::unique_ptr<VT, GenericDeleter<1>>;
    static_assert(!cuda::std::is_constructible<U1, U2&&>::value, "");
  }
  { // Test that if the destination deleter is a reference type then only
    // exact matches are allowed.
    using U1 = cuda::std::unique_ptr<VT, GenericConvertingDeleter<0> const&>;
    using U2 = cuda::std::unique_ptr<VT, GenericConvertingDeleter<0>>;
    using U3 = cuda::std::unique_ptr<VT, GenericConvertingDeleter<0>&>;
    using U4 = cuda::std::unique_ptr<VT, GenericConvertingDeleter<1>>;
    using U5 = cuda::std::unique_ptr<VT, GenericConvertingDeleter<1> const&>;
    static_assert(!cuda::std::is_constructible<U1, U2&&>::value, "");
    static_assert(!cuda::std::is_constructible<U1, U3&&>::value, "");
    static_assert(!cuda::std::is_constructible<U1, U4&&>::value, "");
    static_assert(!cuda::std::is_constructible<U1, U5&&>::value, "");

    using U1C = cuda::std::unique_ptr<const VT, GenericConvertingDeleter<0> const&>;
    static_assert(cuda::std::is_nothrow_constructible<U1C, U1&&>::value, "");
  }
  { // Test that non-reference destination deleters can be constructed
    // from any source deleter type with a suitable conversion. Including
    // reference types.
    using U1 = cuda::std::unique_ptr<VT, GenericConvertingDeleter<0>>;
    using U2 = cuda::std::unique_ptr<VT, GenericConvertingDeleter<0>&>;
    using U3 = cuda::std::unique_ptr<VT, GenericConvertingDeleter<0> const&>;
    using U4 = cuda::std::unique_ptr<VT, GenericConvertingDeleter<1>>;
    using U5 = cuda::std::unique_ptr<VT, GenericConvertingDeleter<1>&>;
    using U6 = cuda::std::unique_ptr<VT, GenericConvertingDeleter<1> const&>;
    static_assert(cuda::std::is_constructible<U1, U2&&>::value, "");
    static_assert(cuda::std::is_constructible<U1, U3&&>::value, "");
    static_assert(cuda::std::is_constructible<U1, U4&&>::value, "");
    static_assert(cuda::std::is_constructible<U1, U5&&>::value, "");
    static_assert(cuda::std::is_constructible<U1, U6&&>::value, "");
  }
}

template <bool IsArray>
__host__ __device__ TEST_CONSTEXPR_CXX23 void test_noexcept()
{
  typedef typename cuda::std::conditional<IsArray, A[], A>::type VT;
  {
    typedef cuda::std::unique_ptr<const VT> APtr;
    typedef cuda::std::unique_ptr<VT> BPtr;
    static_assert(cuda::std::is_nothrow_constructible<APtr, BPtr>::value, "");
  }
  {
    typedef cuda::std::unique_ptr<const VT, CDeleter<const VT>> APtr;
    typedef cuda::std::unique_ptr<VT, CDeleter<VT>> BPtr;
    static_assert(cuda::std::is_nothrow_constructible<APtr, BPtr>::value, "");
  }
  {
    typedef cuda::std::unique_ptr<const VT, NCDeleter<const VT>&> APtr;
    typedef cuda::std::unique_ptr<VT, NCDeleter<const VT>&> BPtr;
    static_assert(cuda::std::is_nothrow_constructible<APtr, BPtr>::value, "");
  }
  {
    typedef cuda::std::unique_ptr<const VT, const NCConstDeleter<const VT>&> APtr;
    typedef cuda::std::unique_ptr<VT, const NCConstDeleter<const VT>&> BPtr;
    static_assert(cuda::std::is_nothrow_constructible<APtr, BPtr>::value, "");
  }
}

template <bool IsArray>
__host__ __device__ TEST_CONSTEXPR_CXX23 void test_deleter_value_category()
{
  typedef typename cuda::std::conditional<IsArray, A[], A>::type VT;
  using TD1 = TrackingDeleter<1>;
  using TD2 = TrackingDeleter<2>;
  TD1 d1;
  TD2 d2;

  { // Test non-reference deleter conversions
    using U1 = cuda::std::unique_ptr<VT, TD1>;
    using U2 = cuda::std::unique_ptr<VT, TD2>;
    U2 u2;
    u2.get_deleter().reset();
    U1 u1(cuda::std::move(u2));
    assert(checkArg<TD2&&>(u1.get_deleter()));
  }
  { // Test assignment from non-const ref
    using U1 = cuda::std::unique_ptr<VT, TD1>;
    using U2 = cuda::std::unique_ptr<VT, TD2&>;
    U2 u2(nullptr, d2);
    U1 u1(cuda::std::move(u2));
    assert(checkArg<TD2&>(u1.get_deleter()));
  }
  { // Test assignment from const ref
    using U1 = cuda::std::unique_ptr<VT, TD1>;
    using U2 = cuda::std::unique_ptr<VT, TD2 const&>;
    U2 u2(nullptr, d2);
    U1 u1(cuda::std::move(u2));
    assert(checkArg<TD2 const&>(u1.get_deleter()));
  }
}

__host__ __device__ TEST_CONSTEXPR_CXX23 bool test()
{
  {
    test_sfinae</*IsArray*/ false>();
    test_noexcept<false>();
    if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
    {
      test_deleter_value_category<false>();
    }
  }
  {
    test_sfinae</*IsArray*/ true>();
    test_noexcept<true>();
    if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
    {
      test_deleter_value_category<true>();
    }
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2023
  static_assert(test());
#endif // TEST_STD_VER >= 2023

  return 0;
}
