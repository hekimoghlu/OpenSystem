/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 1, 2025.
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
#define REF 1
#define CREF 2
#define PTR 3
#define BPTR 4

#if defined(__CCT_TEST_ENABLED)
#define __CCT_ENABLE_USER_SPACE
#endif

#include <sys/constrained_ctypes.h>
#include <darwintest.h>
#include <string.h>
#include <unistd.h>

T_GLOBAL_META(
	T_META_RADAR_COMPONENT_NAME("xnu"),
	T_META_RADAR_COMPONENT_VERSION("networking"),
	T_META_RUN_CONCURRENTLY(TRUE));

/*
 * Verify that `int_ref_t' and `int_ref_ref_t'
 * behave as expected, under different combinations
 * of the CCT being enabled / enacted
 */
#if defined(__CCT_TEST_ENABLED) && defined(__CCT_TEST_ENACTED)
/*
 * When the CCT are enabled in the user space,
 * __CCT_DECLARE_CONSTRAINED_PTR_TYPES(int, int) should
 * define the types `int_ref_t' and `int_ref_ref_t'.
 *
 * If the CCT are enacted in addition to being enabled,
 * the test code itself has to adhere to the type
 * constraints.
 */
__CCT_DECLARE_CONSTRAINED_PTR_TYPES(int, int);

T_DECL(enacted_constrained_types_are_size_compatible_with_plain_types, "rdar://101223528")
{
	T_ASSERT_EQ(sizeof(int_ref_t), sizeof(int*), NULL);
	T_ASSERT_EQ(sizeof(int_ref_ref_t), sizeof(int **), NULL);
}

T_DECL(enacted_constrained_types_are_assignable_from_plain_types, "rdar://101223528")
{
	int s = 1;
	T_ASSERT_EQ(s, 1, NULL);

	int * ps __single = &s;
	int_ref_t rs = ps;
	T_ASSERT_EQ(*rs, 1, NULL);

	int * * pps = &ps;
	int_ref_ref_t rrs = pps;
	T_ASSERT_EQ(**rrs, 1, NULL);
}
#elif defined(__CCT_TEST_ENABLED) && !defined(__CCT_TEST_ENACTED)
/*
 * When the CCT are enabled in the user space,
 * __CCT_DECLARE_CONSTRAINED_PTR_TYPES(int, int) should
 * define the types `int_ref_t' and `int_ref_ref_t'.
 *
 *  When CCT are not enacted, the test code itself does not have to adhere
 * to the type constraints.
 */
__CCT_DECLARE_CONSTRAINED_PTR_TYPES(int, int);

T_DECL(enabled_constrained_types_are_size_compatible_with_plain_types, "rdar://101223528")
{
	T_ASSERT_EQ(sizeof(int_ref_t), sizeof(int*), NULL);
	T_ASSERT_EQ(sizeof(int_ref_ref_t), sizeof(int **), NULL);
}

/*
 *  When CCT are not enacted, the test code itself does not have to adhere
 * to the type constraints.
 */
T_DECL(enabled_constrained_types_are_assignable_from_plain_types, "rdar://101223528")
{
	int s = 1;
	T_ASSERT_EQ(s, 1, NULL);

	int * ps __single = &s;
	int_ref_t rs = ps;
	T_ASSERT_EQ(*rs, 1, NULL);

	int * * pps = &ps;
	int_ref_ref_t rrs = pps;
	T_ASSERT_EQ(**rrs, 1, NULL);
}
#else /* !defined(__CCT_TEST_ENABLED) && !defined(__CCT_TEST_ENACTED) */
/*
 * When the CCT are disabled in the user space,
 * attempt to define `int_ref_t' and `int_ref_ref_t'
 * should be a no-op, and the subsequent redefintion
 * of `int_ref_t' and `int_ref_ref_t' should succeed.
 */
__CCT_DECLARE_CONSTRAINED_PTR_TYPES(int, int);

typedef int * int_ref_t;
typedef int * * int_ref_ref_t;

T_DECL(disabled_constrained_types_decay_into_plain_types, "rdar://101223528")
{
	int s = 1;
	T_ASSERT_EQ(s, 1, NULL);

	int_ref_t rs = &s;
	T_ASSERT_EQ(*rs, 1, NULL);

	int_ref_ref_t rrs = &rs;
	T_ASSERT_EQ(**rrs, 1, NULL);
}
#endif /* !defined(__CCT_TEST_ENABLED) && !defined(__CCT_TEST_ENACTED) */
