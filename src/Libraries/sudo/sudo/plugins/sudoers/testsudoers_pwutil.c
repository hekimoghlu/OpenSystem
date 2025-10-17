/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 31, 2025.
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
#define sudo_make_pwitem	testsudoers_make_pwitem
#define sudo_make_gritem	testsudoers_make_gritem
#define sudo_make_gidlist_item	testsudoers_make_gidlist_item
#define sudo_make_grlist_item	testsudoers_make_grlist_item

#define getpwnam		testsudoers_getpwnam
#define getpwuid		testsudoers_getpwuid
#define getgrnam		testsudoers_getgrnam
#define getgrgid		testsudoers_getgrgid
#define sudo_getgrouplist2_v1	testsudoers_getgrouplist2_v1

#include "tsgetgrpw.h"
#include "pwutil_impl.c"
