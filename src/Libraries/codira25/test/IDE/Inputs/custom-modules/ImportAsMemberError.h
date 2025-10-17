/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 26, 2024.
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

#ifndef IMPORT_AS_MEMBER_ERR_H
#define IMPORT_AS_MEMBER_ERR_H

struct __attribute__((language_name("ErrorStruct"))) IAMStruct {
  double x, y, z;
};

@import Foundation;

@protocol ImportedProtocolBase;
@protocol ImportedProtocolBase <NSObject>
@end
typedef NSObject<ImportedProtocolBase> *ImportedProtocolBase_t;

// Non-prototype declaration
extern void IAMErrorStructHasPrototype(void)
    __attribute__((language_name("ErrorStruct.hasPrototype()"))); // ok

#endif // IMPORT_AS_MEMBER_ERR_H
