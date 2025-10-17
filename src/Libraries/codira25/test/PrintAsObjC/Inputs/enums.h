/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 11, 2022.
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

// This file is meant to be used with the mock SDK, not the real one.
#import <Foundation.h>

#define LANGUAGE_NAME(X) __attribute__((language_name(#X)))

@interface Wrapper : NSObject
@end

enum TopLevelRaw { TopLevelRawA };
enum MemberRaw { MemberRawA } LANGUAGE_NAME(Wrapper.Raw);

typedef enum { TopLevelAnonA } TopLevelAnon;
typedef enum { MemberAnonA } MemberAnon LANGUAGE_NAME(Wrapper.Anon);
typedef enum LANGUAGE_NAME(Wrapper.Anon2) { MemberAnon2A } MemberAnon2;

typedef enum TopLevelTypedef { TopLevelTypedefA } TopLevelTypedef;
typedef enum LANGUAGE_NAME(Wrapper.Typedef) MemberTypedef { MemberTypedefA } MemberTypedef;

typedef NS_ENUM(long, TopLevelEnum) { TopLevelEnumA };
typedef NS_ENUM(long, MemberEnum) { MemberEnumA } LANGUAGE_NAME(Wrapper.Enum);

typedef NS_OPTIONS(long, TopLevelOptions) { TopLevelOptionsA = 1 };
typedef NS_OPTIONS(long, MemberOptions) { MemberOptionsA = 1} LANGUAGE_NAME(Wrapper.Options);
