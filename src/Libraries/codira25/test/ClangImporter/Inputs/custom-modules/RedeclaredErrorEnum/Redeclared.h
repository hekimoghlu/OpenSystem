/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 8, 2022.
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

@import Foundation;
#ifndef NO_IMPORT_BASE_FROM_REDECLARED
@import Base;
#endif

extern NSString * const SomeErrorDomain;
// typedef NS_ERROR_ENUM(SomeErrorDomain, SomeErrorCode);
typedef enum SomeErrorCode : long SomeErrorCode;
enum __attribute__((ns_error_domain(SomeErrorDomain))) SomeErrorCode : long
#ifdef NO_IMPORT_BASE_FROM_REDECLARED
{
  SomeErrorX,
  SomeErrorY
}
#endif
;
