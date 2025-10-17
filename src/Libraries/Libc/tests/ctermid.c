/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 5, 2024.
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

#include <stdio.h>

#include <darwintest.h>

T_DECL(ctermid, "ctermid")
{
  char term[L_ctermid] = { '\0' };
  char *ptr = ctermid(term);
  T_EXPECT_EQ((void*)term, (void*)ptr, "ctermid should return the buffer it received");
  T_EXPECT_GT(strlen(ptr), 0ul, "the controlling terminal should have a name");
}

T_DECL(ctermid_null, "ctermid(NULL)")
{
  char *ptr = ctermid(NULL);
  T_EXPECT_GT(strlen(ptr), 0ul, "the controlling terminal should have a name");
}
