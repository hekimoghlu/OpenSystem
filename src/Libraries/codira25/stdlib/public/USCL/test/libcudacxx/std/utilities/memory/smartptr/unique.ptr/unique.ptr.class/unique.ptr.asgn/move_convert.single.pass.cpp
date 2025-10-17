/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 19, 2022.
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
// <memory>

// unique_ptr

// Test unique_ptr converting move assignment

#include <uscl/std/__memory_>
#include <uscl/std/cassert>
#include <uscl/std/utility>

#include "deleter_types.h"
#include "test_macros.h"
#include "unique_ptr_test_helper.h"

template <class APtr, class BPtr>
__host__ __device__ TEST_CONSTEXPR_CXX23 void testAssign(APtr& aptr, BPtr& bptr)
{
  A* p = bptr.get();
  if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
  {
    assert(A_count == 2);
  }
  aptr = cuda::std::move(bptr);
  assert(aptr.get() == p);
  assert(bptr.get() == 0);
  if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
  {
    assert(A_count == 1);
    assert(B_count == 1);
  }
}

template <class LHS, class RHS>
__host__ __device__ TEST_CONSTEXPR_CXX23 void checkDeleter(LHS& lhs, RHS& rhs, int LHSState, int RHSState)
{
  assert(lhs.get_deleter().state() == LHSState);
  assert(rhs.get_deleter().state() == RHSState);
}

template <class T>
struct NCConvertingDeleter
{
  TEST_CONSTEXPR_CXX23 NCConvertingDeleter()                      = default;
  NCConvertingDeleter(NCConvertingDeleter const&)                 = delete;
  TEST_CONSTEXPR_CXX23 NCConvertingDeleter(NCConvertingDeleter&&) = default;

  template <class U>
  __host__ __device__ TEST_CONSTEXPR_CXX23 NCConvertingDeleter(NCConvertingDeleter<U>&&)
  {}

  __host__ __device__ TEST_CONSTEXPR_CXX23 void operator()(T*) const {}
};

template <class T>
struct NCConvertingDeleter<T[]>
{
  TEST_CONSTEXPR_CXX23 NCConvertingDeleter()                      = default;
  NCConvertingDeleter(NCConvertingDeleter const&)                 = delete;
  TEST_CONSTEXPR_CXX23 NCConvertingDeleter(NCConvertingDeleter&&) = default;

  template <class U>
  __host__ __device__ TEST_CONSTEXPR_CXX23 NCConvertingDeleter(NCConvertingDeleter<U>&&)
  {}

  __host__ __device__ TEST_CONSTEXPR_CXX23 void operator()(T*) const {}
};

struct NCGenericDeleter
{
  TEST_CONSTEXPR_CXX23 NCGenericDeleter()                   = default;
  NCGenericDeleter(NCGenericDeleter const&)                 = delete;
  TEST_CONSTEXPR_CXX23 NCGenericDeleter(NCGenericDeleter&&) = default;

  __host__ __device__ TEST_CONSTEXPR_CXX23 void operator()(void*) const {}
};

__host__ __device__ TEST_CONSTEXPR_CXX23 void test_sfinae()
{
  using DA  = NCConvertingDeleter<A>; // non-copyable deleters
  using DB  = NCConvertingDeleter<B>;
  using UA  = cuda::std::unique_ptr<A>;
  using UB  = cuda::std::unique_ptr<B>;
  using UAD = cuda::std::unique_ptr<A, DA>;
  using UBD = cuda::std::unique_ptr<B, DB>;
  { // cannot move from an lvalue
    static_assert(cuda::std::is_assignable<UA, UB&&>::value, "");
    static_assert(!cuda::std::is_assignable<UA, UB&>::value, "");
    static_assert(!cuda::std::is_assignable<UA, const UB&>::value, "");
  }
  { // cannot move if the deleter-types cannot convert
    static_assert(cuda::std::is_assignable<UAD, UBD&&>::value, "");
    static_assert(!cuda::std::is_assignable<UAD, UB&&>::value, "");
    static_assert(!cuda::std::is_assignable<UA, UBD&&>::value, "");
  }
  { // cannot move-convert with reference deleters of different types
    using UA1 = cuda::std::unique_ptr<A, DA&>;
    using UB1 = cuda::std::unique_ptr<B, DB&>;
    static_assert(!cuda::std::is_assignable<UA1, UB1&&>::value, "");
  }
  { // cannot move-convert with reference deleters of different types
    using UA1 = cuda::std::unique_ptr<A, const DA&>;
    using UB1 = cuda::std::unique_ptr<B, const DB&>;
    static_assert(!cuda::std::is_assignable<UA1, UB1&&>::value, "");
  }
  { // cannot move-convert from unique_ptr<Array[]>
    using UA1 = cuda::std::unique_ptr<A>;
    using UA2 = cuda::std::unique_ptr<A[]>;
    using UB1 = cuda::std::unique_ptr<B[]>;
    static_assert(!cuda::std::is_assignable<UA1, UA2&&>::value, "");
    static_assert(!cuda::std::is_assignable<UA1, UB1&&>::value, "");
  }
  { // cannot move-convert from unique_ptr<Array[]>
    using UA1 = cuda::std::unique_ptr<A, NCGenericDeleter>;
    using UA2 = cuda::std::unique_ptr<A[], NCGenericDeleter>;
    using UB1 = cuda::std::unique_ptr<B[], NCGenericDeleter>;
    static_assert(!cuda::std::is_assignable<UA1, UA2&&>::value, "");
    static_assert(!cuda::std::is_assignable<UA1, UB1&&>::value, "");
  }
}

__host__ __device__ TEST_CONSTEXPR_CXX23 bool test()
{
  test_sfinae();
  {
    cuda::std::unique_ptr<B> bptr(new B);
    cuda::std::unique_ptr<A> aptr(new A);
    testAssign(aptr, bptr);
  }
  if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
  {
    assert(A_count == 0);
    assert(B_count == 0);
  }
  {
    Deleter<B> del(42);
    cuda::std::unique_ptr<B, Deleter<B>> bptr(new B, cuda::std::move(del));
    cuda::std::unique_ptr<A, Deleter<A>> aptr(new A);
    testAssign(aptr, bptr);
    checkDeleter(aptr, bptr, 42, 0);
  }
  if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
  {
    assert(A_count == 0);
    assert(B_count == 0);
  }
  {
    CDeleter<A> adel(6);
    CDeleter<B> bdel(42);
    cuda::std::unique_ptr<B, CDeleter<B>&> bptr(new B, bdel);
    cuda::std::unique_ptr<A, CDeleter<A>&> aptr(new A, adel);
    testAssign(aptr, bptr);
    checkDeleter(aptr, bptr, 42, 42);
  }
  if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
  {
    assert(A_count == 0);
    assert(B_count == 0);
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
