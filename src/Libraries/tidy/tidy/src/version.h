/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 7, 2023.
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
/* Apple Inc. Changes:
   2007-01-29 iccir Added Apple-specific build information
   2016-02-25 ddkilzer FIXME: Need to set TIDY_APPLE_BUILD_NUMBER[STR] macros
              in Visual C++ project based on $(RC_ProjectSourceVersion)
              environment variable for Windows.
*/
#ifdef __APPLE__
static const char TY_(release_date)[] = "31 October 2006" " - Apple Inc. build " TIDY_APPLE_BUILD_NUMBER_STR;
#else
static const char TY_(release_date)[] = "31 October 2006";
#endif
