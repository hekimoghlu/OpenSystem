/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 8, 2023.
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

#include <stdlib.h>

#include <darwintest.h>

T_DECL(setenv_getenv, "getenv returns value set by setenv")
{
	char *name = "foo";
	char *value = "bar";
	int setenv_rc = setenv(name, value, 0);
	T_EXPECT_EQ(0, setenv_rc, "setenv must succeed with 0 return code");
	char *getenv_result = getenv(name);
	T_EXPECT_EQ_STR(value, getenv_result, "getenv must return setenv argument");
}

T_DECL(setenv_overwrite, "getenv returns the latest setenv argument")
{
	char *name = "foo";
	char *first_value = "bar";
	char *second_value = "baz";
	int setenv_rc = 0;
	setenv_rc = setenv(name, first_value, 0);
	T_EXPECT_EQ(0, setenv_rc, "setenv must succeed with 0 return code");
	setenv_rc = setenv(name, second_value, 1);
	T_EXPECT_EQ(0, setenv_rc, "setenv must succeed with 0 return code");
	char *getenv_result = getenv(name);
	T_EXPECT_EQ_STR(second_value, getenv_result, "getenv must return the latest setenv argument");
}

T_DECL(setenv_dont_overwrite, "setenv respects overwrite")
{
	char *name = "foo";
	char *first_value = "bar";
	char *second_value = "baz";
	int setenv_rc = 0;
	setenv_rc = setenv(name, first_value, 0);
	T_EXPECT_EQ(0, setenv_rc, "setenv must succeed with 0 return code");
	setenv_rc = setenv(name, second_value, 0);
	T_EXPECT_EQ(0, setenv_rc, "setenv must succeed with 0 return code");
	char *getenv_result = getenv(name);
	T_EXPECT_EQ_STR(first_value, getenv_result, "the second setenv must not overwrite the first one");
}
/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 8, 2024.
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
 */T_DECL(setenv_accepts_leading_eq_sign, "setenv accepts values starting with '='")
{
	char *name = "foo";
	char *value = "=bar";
	int setenv_rc = setenv(name, value, 0);
	T_EXPECT_EQ(0, setenv_rc, "setenv must succeed with 0 return code");
	char *getenv_result = getenv(name);
	T_EXPECT_EQ_STR(value, getenv_result, "getenv must return setenv argument");
}

T_DECL(setenv_accepts_leading_eq_sign_overwrite, "setenv accepts values starting with '=' when overwriting an existing value")
{
	char *name = "foo";
	char *first_value = "bar";
	char *second_value = "=baz";
	int setenv_rc = 0;
	setenv_rc = setenv(name, first_value, 0);
	T_EXPECT_EQ(0, setenv_rc, "setenv must succeed with 0 return code");
	setenv_rc = setenv(name, second_value, 1);
	T_EXPECT_EQ(0, setenv_rc, "setenv must succeed with 0 return code");
	char *getenv_result = getenv(name);
	T_EXPECT_EQ_STR(second_value, getenv_result, "getenv must return the latest setenv argument");
}

