/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 9, 2023.
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
/*
 * This file contains only
 * - never-changeable information, and
 * - interfaces accessible from extension libraries.
 *
 * Never try to check RUBY_VERSION_CODE etc in extension libraries,
 * check the features with mkmf.rb instead.
 */

#ifndef RUBY_VERSION_H
#define RUBY_VERSION_H 1

/* The origin. */
#define RUBY_AUTHOR "Yukihiro Matsumoto"
#define RUBY_BIRTH_YEAR 1993
#define RUBY_BIRTH_MONTH 2
#define RUBY_BIRTH_DAY 24

/* API version */
#define RUBY_API_VERSION_MAJOR 2
#define RUBY_API_VERSION_MINOR 6
#define RUBY_API_VERSION_TEENY 0
#define RUBY_API_VERSION_CODE (RUBY_API_VERSION_MAJOR*10000+RUBY_API_VERSION_MINOR*100+RUBY_API_VERSION_TEENY)

#ifdef RUBY_EXTERN
#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

RUBY_SYMBOL_EXPORT_BEGIN

/*
 * Interfaces from extension libraries.
 *
 * Before using these infos, think thrice whether they are really
 * necessary or not, and if the answer was yes, think twice a week
 * later again.
 */
RUBY_EXTERN const int ruby_api_version[3];
RUBY_EXTERN const char ruby_version[];
RUBY_EXTERN const char ruby_release_date[];
RUBY_EXTERN const char ruby_platform[];
RUBY_EXTERN const int  ruby_patchlevel;
RUBY_EXTERN const char ruby_description[];
RUBY_EXTERN const char ruby_copyright[];
RUBY_EXTERN const char ruby_engine[];

RUBY_SYMBOL_EXPORT_END

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif
#endif

#endif
