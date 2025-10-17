/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 18, 2023.
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

#include <TargetConditionals.h>
#include <errno.h>
#include <locale.h>
#include <wchar.h>

#include <darwintest.h>

T_DECL(wcscoll_PR_142386677, "Test failed wcscoll() assertion",
    T_META_ENABLED(TARGET_OS_OSX))
{
	int ret;

	setlocale(LC_COLLATE, "en_US.UTF-8");
	ret = wcscoll(L"á»Ÿ", L"á»Ÿ");
	T_ASSERT_EQ(ret, 0, NULL);
}
